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
from tensorflow_privacy.privacy.analysis.compute_dp_sgd_privacy_lib import compute_dp_sgd_privacy
from tqdm import tqdm
import private_CNN
import time
from torch.profiler import profile, record_function, ProfilerActivity


def train_private_functorch(args, model, trainloader, criterion, optimizer, device):
    """
    Train model in a differentially private manner by clipping each per-sample gradient and adding noises.
    """
    model.train()
    name = model.get_name()
    prof = profile(
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=100),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(f'./dp_log/{name}'),
            activities=[ProfilerActivity.CUDA], 
            record_shapes=True
            )
    
    prof.start()
    for i, data in enumerate(tqdm(trainloader)):
        inputs, targets = data
        inputs = inputs.to(device)
        targets = targets.to(device)

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
        
        # 2. clip each per-sample gradient.
        batch_clip(model, args.max_grad_norm)

        # 3. noising and scale.
        batch_noising_scale(model, args.max_grad_norm, args.noise_multiplier, args.batch_size)

        optimizer.step()
        optimizer.zero_grad()
        prof.step()
    
    prof.stop()
    # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    print(prof.key_averages(group_by_stack_n=5).table(sort_by="self_cuda_time_total", row_limit=2))
    # prof.export_chrome_trace(f"./runtime_profiler_results/dp_{name}_trace.json")


def train_private_mixed_ghost(args, model, trainloader, criterion, optimizer, device):
    mixed_ghost_timing = []
    model.train()
    # n_acc_steps = args.batch_size // args.mini_batch_size
    for batch_idx, (inputs, targets) in enumerate(tqdm(trainloader)):
        batch_start_time = time.perf_counter()
        inputs, targets = inputs.to(device), targets.to(device)
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
    opacus_timing = []
    model.train()
    for batch_idx, (images, target) in enumerate(tqdm(trainloader)):
        batch_start_time = time.perf_counter()
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
        batch_end_time = time.perf_counter()
        opacus_timing.append(batch_end_time - batch_start_time)
    return np.mean(opacus_timing)


def train_public(model, trainloader, criterion, optimizer, device):
    public_timing = []
    model.train()
    for batch_idx, (images, target) in enumerate(tqdm(trainloader)):
        batch_start_time = time.perf_counter()
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
        batch_end_time = time.perf_counter()
        public_timing.append(batch_end_time - batch_start_time)
    return np.mean(public_timing)
   

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
        default=1,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )

    parser.add_argument(
        "-b",
        "--batch_size",
        default=100,
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

    trainset = torchvision.datasets.CIFAR10(root='./datasets', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                            shuffle=True)

    testset = torchvision.datasets.CIFAR10(root='./datasets', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                            shuffle=False)

    # classes = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)

    USE_CUDA = torch.cuda.is_available()
    device = torch.device("cuda" if USE_CUDA else "cpu")
    print(f'training device: {device}')
    print(f'training set size: {len(trainset)}')

    # model = models.LargerConvNet(10)
    model = models.SimpleConv()
    # model = models.SampleConvNet()
    # model = models.VGG11(in_channels=3, num_classes=10)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    criterion = nn.CrossEntropyLoss()
    # timings = []
    # if args.mode == 'mixed_ghost':
    #     criterion = nn.CrossEntropyLoss(reduction="none")
    #     privacy_engine = private_CNN.PrivacyEngine(
    #     model,
    #     batch_size=args.batch_size,
    #     sample_size=len(trainloader.dataset),
    #     noise_multiplier=args.noise_multiplier,
    #     epochs=args.epochs,
    #     max_grad_norm=args.max_grad_norm,
    #     ghost_clipping=False,
    #     mixed=False
    #     )
    #     privacy_engine.attach(optimizer)
    #     for epoch in range(1, args.epochs + 1):
    #         result_timing = train_private_mixed_ghost(args, model, trainloader, criterion, optimizer, device)
    #         timings.append(result_timing)
    
    # elif args.mode == 'opacus_dp':
    #     criterion = nn.CrossEntropyLoss()
    #     privacy_engine = PrivacyEngine(
    #         secure_mode=args.secure_rng,
    #     )
    #     clipping = "per_layer" if args.clip_per_layer else "flat"
    #     model, optimizer, train_loader = privacy_engine.make_private(
    #         module=model,
    #         optimizer=optimizer,
    #         data_loader=trainloader,
    #         noise_multiplier=args.noise_multiplier,
    #         max_grad_norm=args.max_grad_norm,
    #         clipping=clipping,
    #     )
    #     for epoch in range(1, args.epochs + 1):
    #         result_timing = train_private_opacus(args, model, trainloader, criterion, optimizer, device)
    #         timings.append(result_timing)

    # elif args.mode == 'functorch_dp':
    #     criterion = nn.CrossEntropyLoss()
    #     for epoch in range(1, args.epochs + 1):
    #         result_timing = train_private_functorch(args, model, trainloader, criterion, optimizer, device)
    #         timings.append(result_timing)
    
    # elif args.mode == 'public':
    #     criterion = nn.CrossEntropyLoss()
    #     for epoch in range(1, args.epochs + 1):
    #         result_timing = train_public(model, trainloader, criterion, optimizer, device)
    #         timings.append(result_timing)

    # print(f'{args.mode}, average timing per {args.batch_size}-sized batch: {np.mean(timings)}')

    train_private_functorch(args, model, trainloader, criterion, optimizer, device)

    test(model, device, testloader)

