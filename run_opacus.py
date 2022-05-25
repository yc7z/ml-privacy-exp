import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import argparse
import models
import matplotlib.pyplot as plt
from privacymech import *
import numpy as np
import time
import cProfile, pstats, io
from pstats import SortKey
from opacus import PrivacyEngine
from torch.profiler import profile, record_function, ProfilerActivity

timing = []
timing_opacus = []
timing_private = []


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

            with torch.autograd.profiler.profile(use_cuda=True) as prof:
            # start to monitor function call
                loss.backward()
                optimizer.step()
            print(prof.key_averages().table(sort_by="cuda_time_total"))


            toc = time.perf_counter()
            pr.disable()

            # optimizer.step()

            # pr.disable()
            # toc = time.perf_counter()

            timing.append(toc - tic)

            running_loss += loss
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
        # print(f'epoch: {epoch + 1}, loss: {running_loss}')
    print('finished vanilla training.')
    # torch.save(model.state_dict(), args.weights_path)
    # print('weights saved.')


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

            with torch.autograd.profiler.profile(use_cuda=True) as prof:
                loss.backward()
                optimizer.step()
            print(prof.key_averages().table(sort_by="cuda_time_total"))


            toc = time.perf_counter()
            pr.disable()

            # s = io.StringIO()
            # sortby = SortKey.CUMULATIVE
            # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
            # ps.print_stats()
            # pr.dump_stats("Opacus,flameresult,batch=" + str(args.batch_size) + ".prof")

            # optimizer.step()

            # pr.disable()
            # toc = time.perf_counter()

            timing_opacus.append(toc - tic)

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
    print(f'training device: {device}')
    
    
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
    elif args.version == 'opacus':
        train_opacus(args, net, trainloader, criterion, optimizer, device)
        # print(f'raw timing info: {timing_opacus}')
        print(f'number of epochs: {args.epochs}, total training time: {sum(timing_opacus)}, dataset size: {total_size}')
        print(f'average time taken on each batch: {sum(timing_opacus) / len(timing_opacus)} (opacus)')

    if args.eval:
        eval_classifier(net, args.weights_path, testloader, classes, device)