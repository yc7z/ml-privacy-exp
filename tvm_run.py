from imghdr import tests
from tracemalloc import start
import torch
import torch.nn as nn
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import argparse
import models
from utils_plus import *
import numpy as np
from functorch import vmap, grad, make_functional
from opacus import PrivacyEngine
# from tensorflow_privacy.privacy.analysis.compute_dp_sgd_privacy_lib import compute_dp_sgd_privacy
from tqdm import tqdm
# import private_CNN
import time
from torch.profiler import profile, record_function, ProfilerActivity
from memory_profiler import profile
import tvm 
import tvm.runtime as t
from tvm.contrib import graph_executor




# @profile
def train_private_functorch(args, model, trainloader, criterion, optimizer, device):
    """
    Train model in a differentially private manner by clipping each per-sample gradient and adding noises.
    """
    model.train()
    for i, data in enumerate(trainloader):
        inputs, targets = data
        inputs = inputs.to(device)
        targets = targets.to(device)

         # compute output
        output = model(inputs)
        loss = criterion(output, targets)

        # compute gradient and do SGD step
        # loss.backward()

        # 1. Compute the gradient w.r.t. each model parameter on each sample within a batch.

        func_model, weights = make_functional(model)

        def compute_loss(weights, image, label):
            images = image.unsqueeze(0)
            labels = label.unsqueeze(0)
            output = func_model(weights, images)
            loss = criterion(output, labels)
            return loss
    
        sample_grads = vmap(grad(compute_loss), (None, 0, 0))(weights, inputs, targets)

        for sample_grad, parameter in zip(sample_grads, model.parameters()):
            parameter.grad_sample = sample_grad.detach()
        
        # for parameter in model.parameters():
        #     parameter.grad_sample = torch.zeros(size=torch.Size([args.batch_size])+parameter.shape, dtype=torch.float, device=device)
        
        # 2. clip each per-sample gradient.
        batch_clip(model, args.max_grad_norm)

        # 3. noising and scale.
        batch_noising_scale(model, args.max_grad_norm, args.noise_multiplier, args.batch_size)

        optimizer.step()
        optimizer.zero_grad()
       
        


def train_private_mixed_ghost(args, model, trainloader, criterion, optimizer, device):
    mixed_ghost_timing = []
    model.train()
    # n_acc_steps = args.batch_size // args.mini_batch_size
    for batch_idx, (inputs, targets) in enumerate(tqdm(trainloader)):
        batch_start_time = time.perf_counter()
        # inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # if ((batch_idx + 1) % n_acc_steps == 0) or ((batch_idx + 1) == len(train_loader)):
        #     optimizer.step(loss=loss)
        #     optimizer.zero_grad()
        # else:
        #     optimizer.virtual_step(loss=loss)
        optimizer.step(loss=loss)
        optimizer.zero_grad()
        batch_end_time = time.perf_counter()
        mixed_ghost_timing.append(batch_end_time - batch_start_time)
    return np.mean(mixed_ghost_timing)


def train_private_opacus(args, model, trainloader, criterion, optimizer, device):
    model.train()
    for batch_idx, (images, target) in enumerate(trainloader):
        images = images.to(device)
        target = target.to(device)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # compute gradient and do SGD step
        loss.backward()

        # make sure we take a step after processing the last mini-batch in the
        # epoch to ensure we start the next epoch with a clean state
        optimizer.step()
        optimizer.zero_grad()


def train_public(model, trainloader, criterion, optimizer, device):
    model.train()
    grad_time = 0
    for batch_idx, (images, target) in enumerate(trainloader):
        images = images.to(device)
        target = target.to(device)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # compute gradient and do SGD step
        grad_start = time.perf_counter()
        loss.backward()
        grad_end = time.perf_counter()
        
        # make sure we take a step after processing the last mini-batch in the
        # epoch to ensure we start the next epoch with a clean state
        optimizer.step()
        optimizer.zero_grad()
        
        grad_time += (grad_end - grad_start)
    return grad_time


def tvm_public(model, trainloader, criterion, optimizer, device):
    # model_name = "resnet18"
    # model = getattr(torchvision.models, model_name)(pretrained=True)
    # devce = "cpu"
    model = model.to(device)
    
    model = model.eval()
    input_shape = [1, 3, 32, 32]
    scripted_model = torch.jit.trace(model, torch.randn(input_shape).to(device)).eval()

    # tvm line
    input_name = "input_name"
    shape_list = [(input_name, (1, 3, 32, 32))]
    graph, params = tvm.relay.frontend.from_pytorch(scripted_model, shape_list)
    target =  tvm.target.Target("llvm", host="llvm")
    # dev = tvm.cuda(0)
    dev = tvm.cpu(0)
    with tvm.transform.PassContext(opt_level=3):
        lib = tvm.relay.build(graph, target=target, params=params)

    dtype = "float32"
    m = graph_executor.GraphModule(lib["default"](dev))

    model.train()
    grad_time = 0
    for batch_idx, (images, target) in enumerate(trainloader):
        images = images.to(device)
        target = target.to(device)

        m.set_input(input_name, tvm.nd.array(images))
        m.run()
        tvm_output = m.get_output(0)

        # compute output
        # output = model(images)


        loss = criterion(tvm_output, target)

        # compute gradient and do SGD step
        grad_start = time.perf_counter()
        loss.backward()
        grad_end = time.perf_counter()
        
        # make sure we take a step after processing the last mini-batch in the
        # epoch to ensure we start the next epoch with a clean state
        optimizer.step()
        optimizer.zero_grad()
        
        grad_time += (grad_end - grad_start)
    return grad_time
   

def test(model, device, testloader):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in tqdm(testloader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(testloader.dataset)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
            test_loss,
            correct,
            len(testloader.dataset),
            100.0 * correct / len(testloader.dataset),
        )
    )
    return correct / len(testloader.dataset)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PyTorch CIFAR10 DP Training")

    parser.add_argument(
        "--epochs",
        default=2,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )

    parser.add_argument(
        "-b",
        "--batch_size",
        default=64,
        type=int,
    )
    parser.add_argument(
        "--sample-rate",
        default=0.04,
        type=float,
        metavar="SR",
        help="sample rate used for batch construction (default: 0.005)",
    )
    parser.add_argument(
        "--lr",
        "--learning-rate",
        default=4e-2,
        type=float,
        metavar="LR",
        help="initial learning rate",
        dest="lr",
    )
    parser.add_argument(
        "--momentum", default=0.9, type=float, metavar="M", help="SGD momentum"
    )

    parser.add_argument(
        "--seed", default=None, type=int, help="seed for initializing training. "
    )

    parser.add_argument(
        "--noise_multiplier",
        type=float,
        default=4.0,
        metavar="S",
        help="Noise multiplier (default 1.0)",
    )
    parser.add_argument(
        "-c",
        "--max_grad_norm",
        type=float,
        default=1.0,
        metavar="C",
        help="Clip per-sample gradients to this norm (default 1.0)",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=1e-5,
        metavar="D",
        help="Target delta (default: 1e-5)",
    )
    parser.add_argument(
        '--mode',
        type=str,
        default='functorch_dp'
    )
    parser.add_argument(
        "--clip_per_layer",
        action="store_true",
        default=False,
        help="Use static per-layer clipping with the same clipping threshold for each layer. Necessary for DDP. If `False` (default), uses flat clipping.",
    )
    parser.add_argument(
        "--secure-rng",
        action="store_true",
        default=False,
        help="Enable Secure RNG to have trustworthy privacy guarantees."
        "Comes at a performance cost. Opacus will emit a warning if secure rng is off,"
        "indicating that for production use it's recommender to turn it on.",
    )
    parser.add_argument(
        "--model_name"
    )

    args = parser.parse_args()

    # prepare training & testing data
    transform = transforms.Compose(
    [
        transforms.ToTensor()
    ])

    # transform = transforms.Compose(
    # [transforms.Resize((224, 224)),
    #  transforms.ToTensor(),
    #  transforms.Normalize(mean=(0.5), std=(0.5))])
    USE_CUDA = torch.cuda.is_available()
    # device = torch.device("cuda" if USE_CUDA else "cpu")
    device = torch.device("cpu")
    print(f'training device: {device}')

    trainset = torchvision.datasets.CIFAR10(root='./datasets', train=True,
                                            download=True, transform=transform,
                                            )
    
    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=args.batch_size,
                                            pin_memory=True,
                                            shuffle=True)

    testset = torchvision.datasets.CIFAR10(root='./datasets', train=False,
                                        download=True, transform=transform)

    testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=args.batch_size, 
                                            pin_memory=True,
                                            shuffle=False)
    

    # model = models.LargerConvNet(10)
    # model = models.SimpleConvNet()
    model = models.VGG11(in_channels=3, num_classes=10)
    model.to(device)
    model_name = model.get_name()
    args.model_name = model_name

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if args.mode == 'mixed_ghost':
        # criterion = nn.CrossEntropyLoss(reduction="none")
        # privacy_engine = private_CNN.PrivacyEngine(
        # model,
        # batch_size=args.batch_size,
        # sample_size=len(trainloader.dataset),
        # noise_multiplier=args.noise_multiplier,
        # epochs=args.epochs,
        # max_grad_norm=args.max_grad_norm,
        # ghost_clipping=False,
        # mixed=False
        # )
        # privacy_engine.attach(optimizer)
        # for epoch in range(1, args.epochs + 1):
        #     train_private_mixed_ghost(args, model, trainloader, criterion, optimizer, device)
        pass
    
    elif args.mode == 'opacus_dp':
        criterion = nn.CrossEntropyLoss()
        privacy_engine = PrivacyEngine(
            secure_mode=args.secure_rng,
        )
        clipping = "per_layer" if args.clip_per_layer else "flat"
        model, optimizer, train_loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=trainloader,
            noise_multiplier=args.noise_multiplier,
            max_grad_norm=args.max_grad_norm,
            clipping=clipping,
        )

        start_time = time.perf_counter()
        for epoch in range(1, args.epochs + 1):
            train_private_opacus(args, model, trainloader, criterion, optimizer, device)
        end_time = time.perf_counter()

    elif args.mode == 'functorch_dp':
        criterion = nn.CrossEntropyLoss()
        # total_per_sample_grad_time = 0
        # total_clip_time = 0
        # total_noise_time = 0
        # total_data_batch_time = 0
        # total_optimizer_step_time = 0

        start_time = time.perf_counter()
        for epoch in range(1, args.epochs + 1):
            # per_sample_grad_time, clip_time, noise_time, data_time, opt_time = train_private_functorch(args, model, trainloader, criterion, optimizer, device)
            # total_per_sample_grad_time += per_sample_grad_time
            # total_clip_time += clip_time
            # total_noise_time += noise_time
            # total_data_batch_time += data_time
            # total_optimizer_step_time += opt_time
            train_private_functorch(args, model, trainloader, criterion, optimizer, device)
        end_time = time.perf_counter()
        # print(f'total time for per sample gradient during training: {total_per_sample_grad_time}\n')
        # print(f'total time for clipping during training: {total_clip_time}\n')
        # print(f'total time for noising during training: {total_noise_time}\n')
        # print(f'total time for getting data batch: {total_data_batch_time}\n')
        # print(f'total time for optimizer step: {total_optimizer_step_time}\n')

    
    elif args.mode == 'public':
        criterion = nn.CrossEntropyLoss()
        # total_grad_time = 0
        start_time = time.perf_counter()
        for epoch in range(1, args.epochs + 1):
            grad_time = train_public(model, trainloader, criterion, optimizer, device)
            # total_grad_time += grad_time
        end_time = time.perf_counter()

    elif args.mode == 'public_tvm':
        criterion = nn.CrossEntropyLoss()
        # total_grad_time = 0
        start_time = time.perf_counter()
        for epoch in range(1, args.epochs + 1):
            grad_time = tvm_public(model, trainloader, criterion, optimizer, device)
            # total_grad_time += grad_time
        end_time = time.perf_counter()

    


    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'total number of trainable parameters: {pytorch_total_params}')
    test(model, device, testloader)
    print(f'total training time = {end_time - start_time}s\n')
    
    # print(f'total gradient time: {total_grad_time}s\n')


