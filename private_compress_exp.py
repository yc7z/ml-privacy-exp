import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import argparse
import models
import matplotlib.pyplot as plt
from utils import *
import numpy as np
from torch.profiler import profile, record_function, ProfilerActivity
from functorch import make_functional_with_buffers, vmap, grad



def train_vanilla(args, model, trainloader, criterion, optimizer, device):
    """
        Train model without any privacy mechanisms.
    """
    for epoch in range(args.epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            optimizer.zero_grad()
            loss = criterion(outputs, labels)

            with torch.autograd.profiler.profile(use_cuda=True) as prof:
                loss.backward()
                optimizer.step()
            print(prof.key_averages().table(sort_by="cuda_time_total"))

            running_loss += loss
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')

        print(f'epoch {epoch + 1} finished.')
    print('finished vanilla training.')
    torch.save(model.state_dict(), args.weights_path + '_vanilla' + 'ep' + str(args.epochs) + '.pth')


def train_exp(args, model, trainloader, criterion, device):
    """
    Train model in a differentially private manner by clipping each per-sample gradient and adding noises.
    """

    ft_compute_grad = grad(compute_loss_stateless_model, argnums=2)
    ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=(None, None, None, None, 0, 0))

    print('start training.')
    for epoch in range(1, args.epochs + 1):
        for i, data in enumerate(trainloader):
            inputs, targets = data
            inputs = inputs.to(device)
            targets = targets.to(device)

            # 1. Compute the gradient w.r.t. each model parameter on each sample within a batch.
            fmodel, params, buffers = make_functional_with_buffers(model)
            grads = ft_compute_sample_grad(fmodel, criterion, params, buffers, inputs, targets)

            if args.accumulate_grad and (i + 1) % args.accum_period == 1:
                grad_accumulation = init_accumulation(grads)

            # 2. (optionally) clip, noising, and compress
            if args.clip:
                grads = batch_clip(grads, args.max_grad_norm)
            if args.noising:
                grads = batch_noising(grads, args.max_grad_norm)
            if args.compress:
                grads = topk_compress(grads, args.topk_percentile)
            if args.accumulate_grad:
                for j in range(len(grads)):
                    grad_accumulation[j] += grads[j]

            # 3. Update model parameters via gradient descent.
            with torch.no_grad():
                for param, grad_p in zip(model.parameters(), grads):
                    param -= args.lr * torch.mean(grad_p, dim=0)
                
                # apply the accumulated gradient if required.
                if args.accumulate_grad and i != 0 and i % args.accum_period == 0:
                    for param, grad_accum in zip(model.parameters(), grad_accumulation):
                        param -= args.lr * args.accum_period * torch.mean(grad_accum, dim=0)
    
        print(f'epoch {epoch} finished.')

        # save trained model parameters according to the type of training conducted.
        if not args.noising and not args.compress and not args.clip and not args.accumulate_grad: 
            # vanilla
            torch.save(model.state_dict(), f'{args.weights_path}/vanilla/epochs={epoch}.pth')

        elif args.clip and not args.noising and not args.compress and not args.accumulate_grad: 
            # vanilla except clipping gradients.
            torch.save(model.state_dict(), f'{args.weights_path}/clip_grads/epochs={epoch},clip={args.max_grad_norm}.pth')

        elif args.compress and not args.clip and not args.noising and not args.accumulate_grad:
            # vanilla + top_k compression
            torch.save(model.state_dict(), f'{args.weights_path}/vanilla_topk/epochs={epoch},percentile={args.topk_percentile}.pth')

        elif args.noising and args.clip and not args.compress and not args.accumulate_grad:
            # private sgd via gaussian mechanism, no compression
            torch.save(model.state_dict(), f'{args.weights_path}/gauss_sgd/epochs={epoch},clip={args.max_grad_norm},noise_mult={args.noise_multiplier}.pth')

        elif args.noising and args.compress and args.clip and not args.accumulate_grad:
            # private sgd and top_k compression
            torch.save(model.state_dict(), f'{args.weights_path}/gauss_sgd_topk/epochs={epoch},clip={args.max_grad_norm},noise_mult={args.noise_multiplier},percentile={args.topk_percentile}.pth')

        elif args.noising and args.compress and args.clip and args.accumulate_grad:
            torch.save(model.state_dict(), f'{args.weights_path}/gauss_sgd_topk_accum/epochs={epoch},clip={args.max_grad_norm},noise_mult={args.noise_multiplier},percentile={args.topk_percentile},accumelate_period={args.accum_period}.pth')

    print('finished training.')


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


if __name__ == "__main__":

    # command line arguments for training. mainly hyperparameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=4e-3)
    parser.add_argument('--batch_size', type=int, default=60)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--momentum', type=float, default=0.9)
    # norm threshold for clipping gradients
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    
    parser.add_argument('--weights_path', type=str, default='./weights_by_epochs')
    parser.add_argument('--noise_multiplier', type=float, default=0.3)
    parser.add_argument('--eval', type=bool, default=False)
    parser.add_argument('--topk_percentile', type=float, default=0.005)

    # accumulate gradient every accum_period minibatches.
    parser.add_argument('--accum_period', type=int, default=10)

    # arguments that determine experiments type
    parser.add_argument('--clip', type=bool, default=False)
    parser.add_argument('--noising', type=bool, default=False)
    parser.add_argument('--compress', type=bool, default=False)
    parser.add_argument('--accumulate_grad', type=bool, default=False)

    args = parser.parse_args()

     # prepare training & testing data
    transform = transforms.Compose(
    [
        transforms.ToTensor()
    ])

    trainset = torchvision.datasets.MNIST(root='./datasets', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                            shuffle=True)

    testset = torchvision.datasets.MNIST(root='./datasets', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                            shuffle=False)

    classes = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)

    USE_CUDA = torch.cuda.is_available()
    device = torch.device("cuda" if USE_CUDA else "cpu")
    print(f'training device: {device}')
    print(len(trainloader))
    
    
    net = models.LeNet(10, input_channel=1)
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')
    optimizer = optim.SGD(params=net.parameters(), lr=args.lr, momentum=args.momentum)


    train_exp(args, net, trainloader, criterion, device)