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

timing = []



def train_vanilla(args, model, trainloader, criterion, optimizer):
    """
        Train model without any privacy mechanisms.
    """
    for epoch in range(args.epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader):
            inputs, labels = data
            outputs = model(inputs)

            optimizer.zero_grad()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
        # print(f'epoch: {epoch + 1}, loss: {running_loss}')
    print('finished training.')
    torch.save(model.state_dict(), args.weights_path)
    print('weights saved.')


def eval_classifier(model, weights_path, testloader, classes):
    """
        Load pretrained weights and evaluate model performance.
    """
    model.load_state_dict(torch.load(weights_path))
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    with torch.no_grad():
        for data in testloader:
            images, true_labels = data
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


def train_private(args, model, trainloader, criterion, optimizer):
    """
    Train model in a differentially private manner by clipping the gradients and adding Gaussian noises.
    """
    pr = cProfile.Profile()

    ft_compute_grad = grad(compute_loss_stateless_model, argnums=2)
    ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=(None, None, None, None, 0, 0))
    
    for epoch in range(args.epochs):
        for _, data in enumerate(trainloader):
            inputs, targets = data

            tic = time.perf_counter()
            # start to monitor function call
            pr.enable()

            # 1. Compute the gradient w.r.t. each model parameter on each sample within a batch.
            fmodel, params, buffers = make_functional_with_buffers(model)
            grads = ft_compute_sample_grad(fmodel, criterion, params, buffers, inputs, targets)

            # 2. clipp and noising
            grads = batch_clip(grads, args.max_grad_norm)
            grads = batch_noising(grads, args.max_grad_norm)

            pr.disable()
            toc = time.perf_counter()
            timing.append(toc - tic)

            s = io.StringIO()
            sortby = SortKey.CUMULATIVE
            ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
            ps.print_stats()

            # 3. update model parameters with gradients
            with torch.no_grad():
                for param, grad_p in zip(model.parameters(), grads):
                    param -= args.lr * torch.mean(grad_p, dim=0)
    
        print(f'epoch {epoch + 1} finished.')
    print('finished training.')
    torch.save(model.state_dict(), args.weights_path)
           
                
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
    
    
    net = models.LeNet(10, input_channel=1)
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')
    optimizer = optim.SGD(params=net.parameters(), lr=args.lr, momentum=args.momentum)

    train_private(args, net, trainloader, criterion, optimizer)

    # train_vanilla(args, net, trainloader, criterion, optimizer)

    total_size = len(trainloader.dataset)
    step = int(total_size / args.batch_size)
    print(f'raw timing info: {timing}')
    timing_steps = [sum(timing[i:i+step]) for i in range(0, len(timing), step)]
    print(f'time taken to process each epoch: {timing_steps}')
    print(f'average time taken on each epoch: {sum(timing_steps) / len(timing_steps)}')

    filename = "PrivateTiming,num_microbatch=" + str(step)
    print("Private batch training time when batch = " + str(args.batch_size) + " is ", np.mean(timing))

    with open(filename, "wb") as dill_file:
        dill.dump(timing, dill_file)

    eval_classifier(net, args.weights_path, testloader, classes)