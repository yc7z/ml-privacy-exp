import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import argparse
import models
import matplotlib.pyplot as plt
from functorch import make_functional_with_buffers, vmap, grad
from privacymech import *
import dill
import numpy as np
import time
import cProfile, pstats, io
from pstats import SortKey
from opacus import PrivacyEngine
import jax

timing = []
timing_opacus = []
timing_private = []
timing_jax = []


def train_vanilla(args, model, trainloader, criterion, optimizer, device):
    """
        Train model without any privacy mechanisms.
    """
    pr = cProfile.Profile()
    for epoch in range(args.epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # tic = time.perf_counter()
            # # start to monitor function call
            # pr.enable()
            outputs = model(inputs)

            optimizer.zero_grad()
            loss = criterion(outputs, labels)

            pr.enable()
            tic = time.perf_counter()
            # start to monitor function call
            loss.backward()
            toc = time.perf_counter()
            pr.disable()
            timing.append(toc - tic)

            optimizer.step()

            # pr.disable()
            # toc = time.perf_counter()

            running_loss += loss
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
        # print(f'epoch: {epoch + 1}, loss: {running_loss}')
    print('finished training.')
    torch.save(model.state_dict(), args.weights_path)
    print('weights saved.')


def eval_classifier(model, weights_path, testloader, classes, device):
    """
        Load pretrained weights and evaluate model performance.
    """
    model.load_state_dict(torch.load(weights_path))
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    with torch.no_grad():
        for data in testloader:
            images, true_labels = data
            images = images.to(device)
            true_labels = true_labels.to(device)

            logits = model(images)
            predictions = torch.argmax(logits, dim=1)
            for label, prediction in zip(true_labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

    total_correct = 0
    total = 0
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname} is {accuracy:.1f} %')
        total_correct += correct_count
        total += total_pred[classname]
    
    print(f'Accuracy of the network on the 10000 test images: {100 * total_correct / total:2f} %')


def train_private(args, model, trainloader, criterion, optimizer, device):
    """
    Train model in a differentially private manner by clipping the gradients and adding Gaussian noises.
    """
    # Build-in Python Profiler
    pr = cProfile.Profile()

    # Profiler for Tensorboard
    # active records the rounds 
    # prof = torch.profiler.profile(
    #     schedule=torch.profiler.schedule(wait=1, warmup=1, active=1, repeat=1),
    #     on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/tensorboard1'),
    #     record_shapes=True,
    #     with_stack=True)

    ft_compute_grad = grad(compute_loss_stateless_model, argnums=2)
    ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=(None, None, None, None, 0, 0))
    
    for epoch in range(args.epochs):
        # prof.start()
        for _, data in enumerate(trainloader):
            inputs, targets = data
            inputs = inputs.to(device)
            targets = targets.to(device)

            tic = time.perf_counter()
            # start to monitor function call
            pr.enable()
            # start to monitor function call for tensorboard

            # 1. Compute the gradient w.r.t. each model parameter on each sample within a batch.
            fmodel, params, buffers = make_functional_with_buffers(model)
            grads = ft_compute_sample_grad(fmodel, criterion, params, buffers, inputs, targets)

            # 2. clipp and noising
            grads = batch_clip(grads, args.max_grad_norm)
            grads = batch_noising(grads, args.max_grad_norm)

            # stop to record the profiling
            # pr.disable()
            # prof.step()
            
            # toc = time.perf_counter()
            # timing_private.append(toc - tic)

            # s = io.StringIO()
            # sortby = SortKey.CUMULATIVE
            # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
            # ps.print_stats()
            # pr.dump_stats("PytorchPrivateTiming,train_size=" + str(60000) + ".prof")

            # 3. update model parameters with gradients
            with torch.no_grad():
                for param, grad_p in zip(model.parameters(), grads):
                    param -= args.lr * torch.mean(grad_p, dim=0)
            
            pr.disable()
            toc = time.perf_counter()
            timing_private.append(toc - tic)

            # s = io.StringIO()
            # sortby = SortKey.CUMULATIVE
            # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
            # ps.print_stats()
            # pr.dump_stats("Private,flameresult,batch=" + str(120) + ".prof")
        
        # prof.stop()

    
        print(f'epoch {epoch + 1} finished.')
    print('finished training.')
    # torch.save(model.state_dict(), args.weights_path)


def train_jax(args, model, trainloader, criterion, optimizer, device):
    """
    Train model in a differentially private manner by clipping the gradients and adding Gaussian noises.
    """
    # Build-in Python Profiler
    pr = cProfile.Profile()

    # Profiler for Tensorboard
    # active records the rounds 
    # prof = torch.profiler.profile(
    #     schedule=torch.profiler.schedule(wait=1, warmup=1, active=1, repeat=1),
    #     on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/tensorboard1'),
    #     record_shapes=True,
    #     with_stack=True)

    ft_compute_grad = jax.grad(compute_loss_stateless_model)
    ft_compute_sample_grad = jax.vmap(ft_compute_grad, in_axes=(None, 0, 0, 0))
    perex_grads = jax.jit(ft_compute_sample_grad)
    
    for epoch in range(args.epochs):
        # prof.start()
        for _, data in enumerate(trainloader):
            inputs, targets = data
            inputs = inputs.to(device)
            targets = targets.to(device)

            tic = time.perf_counter()
            # start to monitor function call
            pr.enable()
            # start to monitor function call for tensorboard

            # 1. Compute the gradient w.r.t. each model parameter on each sample within a batch.
            grads = perex_grads(model.parameters(), inputs, targets)

            # 2. clipp and noising
            grads = batch_clip(grads, args.max_grad_norm)
            grads = batch_noising(grads, args.max_grad_norm)

            # stop to record the profiling
            # pr.disable()
            # prof.step()
            
            # toc = time.perf_counter()
            # timing_jax.append(toc - tic)

            # s = io.StringIO()
            # sortby = SortKey.CUMULATIVE
            # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
            # ps.print_stats()
            # pr.dump_stats("PytorchPrivateTiming,train_size=" + str(60000) + ".prof")

            # 3. update model parameters with gradients
            with torch.no_grad():
                for param, grad_p in zip(model.parameters(), grads):
                    param -= args.lr * torch.mean(grad_p, dim=0)
            
            pr.disable()
            toc = time.perf_counter()
            timing_jax.append(toc - tic)

            # s = io.StringIO()
            # sortby = SortKey.CUMULATIVE
            # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
            # ps.print_stats()
            # pr.dump_stats("Private,flameresult,batch=" + str(120) + ".prof")
        
        # prof.stop()

    
        print(f'epoch {epoch + 1} finished.')
    print('finished training.')
    # torch.save(model.state_dict(), args.weights_path)


def train_opacus(args, model, train_loader, criterion, optimizer, device):
    privacy_engine = PrivacyEngine()

    model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
    module=model,
    optimizer=optimizer,
    data_loader=train_loader,
    epochs=args.epochs,
    target_epsilon=args.epsilon,
    target_delta=args.delta,
    max_grad_norm=args.max_grad_norm,
    )

    pr = cProfile.Profile()

    for epoch in range(1, args.epochs + 1):
        running_loss = 0.0
        for i, data in enumerate(train_loader):
            inputs, targets = data
            inputs = inputs.to(device)
            targets = targets.to(device)

            # tic = time.perf_counter()
            # pr.enable()

            outputs = model(inputs)

            optimizer.zero_grad()
            loss = criterion(outputs, targets)

            pr.enable()
            tic = time.perf_counter()
            loss.backward()
            toc = time.perf_counter()
            pr.disable()
            timing_opacus.append(toc - tic)

            optimizer.step()

            # pr.disable()
            # toc = time.perf_counter()
            

            running_loss += loss
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
        print(f'finished epoch {epoch}')
    print(f'finished opacus training.')


if __name__ == "__main__":

    # command line arguments for training. mainly hyperparameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=4e-3)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--momentum', type=float, default=0.9)
    # norm threshold for clipping gradients
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--weights_path', type=str, default='./saved_weights/cifar_LeNet.pth')
    parser.add_argument('--noise_multiplier', type=float, default=0.3)
    parser.add_argument('--epsilon', type=float, default=0.1)
    parser.add_argument('--delta', type=float, default=0.1)
    parser.add_argument('--version', type=str, default='vanilla')
    parser.add_argument('--eval', type=bool, default=False)

    args = parser.parse_args()

     # prepare training & testing data
    transform = transforms.Compose(
    [
        # transforms.Resize((32, 32)),
        transforms.ToTensor()
    # ,transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.MNIST(root='./datasets', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                            shuffle=True)

    testset = torchvision.datasets.MNIST(root='./datasets', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                            shuffle=False)

    # classes = ('plane', 'car', 'bird', 'cat',
    #         'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    classes = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)

    USE_CUDA = torch.cuda.is_available()
    device = torch.device("cuda" if USE_CUDA else "cpu")
    
    
    net = models.LeNet(10, input_channel=1)
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')
    optimizer = optim.SGD(params=net.parameters(), lr=args.lr, momentum=args.momentum)

    total_size = len(trainloader.dataset)

    if args.version == 'vanilla':
        train_vanilla(args, net, trainloader, criterion, optimizer, device)
        # print(f'raw timing info: {timing}')
        print(f'number of epochs: {args.epochs}, total training time: {sum(timing)}, dataset size: {total_size}')
        print(f'average time taken on each batch: {sum(timing) / len(timing)} (vanilla)')
    elif args.version == 'private':
        train_private(args, net, trainloader, criterion, optimizer, device)
        print(f'number of epochs: {args.epochs}, total training time: {sum(timing_private)}, dataset size: {total_size}')
        print(f'average time taken on each batch: {sum(timing_private) / len(timing_private)} (private)')
    elif args.version == 'opacus':
        train_opacus(args, net, trainloader, criterion, optimizer, device)
        # print(f'raw timing info: {timing_opacus}')
        print(f'number of epochs: {args.epochs}, total training time: {sum(timing_opacus)}, dataset size: {total_size}')
        print(f'average time taken on each batch: {sum(timing_opacus) / len(timing_opacus)} (opacus)')
    elif args.version == 'jax':
        train_jax(args, net, trainloader, criterion, optimizer, device)
        # print(f'raw timing info: {timing_opacus}')
        print(f'number of epochs: {args.epochs}, total training time: {sum(timing_jax)}, dataset size: {total_size}')
        print(f'average time taken on each batch: {sum(timing_jax) / len(timing_jax)} (opacus)')

    # train_vanilla(args, net, trainloader, criterion, optimizer, device)

    # total_size = len(trainloader.dataset)
    # step = int(total_size / args.batch_size)
    # print(f'raw timing info: {timing}')
    # # timing_steps = [sum(timing[i:i+step]) for i in range(0, len(timing), step)]
    # print(f'number of epochs: {args.epochs}, total training time: {sum(timing)}, dataset size: {total_size}')
    # # print(f'time taken to process each epoch: {timing_steps}')
    # # print(f'average time taken on each epoch: {sum(timing_steps) / len(timing_steps)}')

    # print("Average private training time per batch when batch = " + str(args.batch_size) + " is ", np.mean(timing))

    # filename = "PytorchPrivateTiming,batch_size=" + str(args.batch_size)
    # with open(filename, "wb") as dill_file:
    #     dill.dump(timing, dill_file)

    if args.eval:
        eval_classifier(net, args.weights_path, testloader, classes, device)