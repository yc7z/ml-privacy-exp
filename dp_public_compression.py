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
import os
import pickle
from tensorflow_privacy.privacy.analysis.compute_dp_sgd_privacy_lib import compute_dp_sgd_privacy
from tqdm import tqdm
from itertools import cycle


def train(model, train_loader, criterion, optimizer, device, epoch):
    model.train()
    losses = []
    for i, (data, target) in enumerate(tqdm(train_loader)):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    print(f"Train Epoch: {epoch} \t Loss: {np.mean(losses):.6f}")


def train_gauss_sgd(args, model, train_loader, criterion, device):
    """
    Train model in a differentially private manner by clipping each per-sample gradient and adding noises.
    """

    ft_compute_grad = grad(compute_loss_stateless_model, argnums=2)
    ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=(None, None, None, None, 0, 0))

    model.train()
    for i, data in enumerate(tqdm(train_loader)):
        inputs, targets = data
        inputs = inputs.to(device)
        targets = targets.to(device)

        # 1. Compute the gradient w.r.t. each model parameter on each sample within a batch.
        fmodel, params, buffers = make_functional_with_buffers(model)
        grads = ft_compute_sample_grad(fmodel, criterion, params, buffers, inputs, targets)

        with torch.no_grad():
            # 2. clip each per-sample gradient
            grads = batch_clip(grads, args.max_grad_norm)

            # 3. take mean of grads over a batch
            batch_grads = []
            for grad_p in grads:
                batch_grads.append(torch.mean(grad_p, dim=0))
                del grad_p
        
            # 4. add gaussian noise
            batch_grads = batch_noising(batch_grads, clip=args.max_grad_norm, noise_multiplier=args.noise_multiplier)

            # 5. Update model parameters via gradient descent.
            for param, grad_p_b in zip(model.parameters(), batch_grads):
                param -= args.lr * grad_p_b
                del grad_p_b
                del param.grad
            model.zero_grad()


def train_gauss_sgd_topk(args, model, train_loader, criterion, device):
    """
    Train model in a differentially private manner by clipping each per-sample gradient and adding noises.
    Also compress gradient before update.
    """

    ft_compute_grad = grad(compute_loss_stateless_model, argnums=2)
    ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=(None, None, None, None, 0, 0))

    model.train()
    for i, data in enumerate(tqdm(train_loader)):
        inputs, targets = data
        inputs = inputs.to(device)
        targets = targets.to(device)

        # 1. Compute the gradient w.r.t. each model parameter on each sample within a batch.
        fmodel, params, buffers = make_functional_with_buffers(model)
        grads = ft_compute_sample_grad(fmodel, criterion, params, buffers, inputs, targets)

        with torch.no_grad():
            # 2. clip each per-sample gradient
            grads = batch_clip(grads, args.max_grad_norm)

            # 3. take mean of grads over a batch
            batch_grads = []
            for grad_p in grads:
                batch_grads.append(torch.mean(grad_p, dim=0))
                del grad_p
        
            # 4. add gaussian noise
            batch_grads = batch_noising(batch_grads, clip=args.max_grad_norm, noise_multiplier=args.noise_multiplier)

            batch_grads = topk_compress(batch_grads, args.topk_percentile)

            # 5. Update model parameters via gradient descent.
            for param, grad_p_b in zip(model.parameters(), batch_grads):
                param -= args.lr * grad_p_b
                del grad_p_b
                del param.grad
            model.zero_grad()


def train_public_aid_topk(args, model, train_loader, pub_loader, criterion, device):
    """
    Train model in a differentially private manner by clipping each per-sample gradient and adding noises.
    Compute gradient on public data and use its topk mask to compress privatized gradient.
    """

    ft_compute_grad = grad(compute_loss_stateless_model, argnums=2)
    ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=(None, None, None, None, 0, 0))

    model.train()
    for data, pub_data in zip(tqdm(train_loader), cycle(pub_loader)):
        inputs, targets = data
        inputs = inputs.to(device)
        targets = targets.to(device)
        pub_inputs, pub_targets = pub_data
        pub_inputs, pub_targets = pub_inputs.to(device), pub_targets.to(device)
        
        # 0. compute gradient on public data without updating it.
        pub_loss = criterion(model(pub_inputs), pub_targets)
        pub_grads = torch.autograd.grad(pub_loss, model.parameters())

        pub_masks = topk_mask_all(pub_grads, args.topk_percentile)
        model.zero_grad()

        # 1. Compute the gradient w.r.t. each model parameter on each sample within a batch.
        fmodel, params, buffers = make_functional_with_buffers(model)
        grads = ft_compute_sample_grad(fmodel, criterion, params, buffers, inputs, targets)

        with torch.no_grad():
            # 2. clip each per-sample gradient
            grads = batch_clip(grads, args.max_grad_norm)

            # 3. take mean of grads over a batch
            batch_grads = []
            for grad_p in grads:
                batch_grads.append(torch.mean(grad_p, dim=0))
                del grad_p
        
            # 4. add gaussian noise
            batch_grads = batch_noising(batch_grads, clip=args.max_grad_norm, noise_multiplier=args.noise_multiplier)

            # 5. Update model parameters via gradient descent.
            for param, grad_p_b, pub_mask in zip(model.parameters(), batch_grads, pub_masks):
                param -= args.lr * torch.multiply(grad_p_b, pub_mask)
                del grad_p_b
                del param.grad
            model.zero_grad()


def experiment(args, model, train_loader, criterion, optimizer, device, test_loader, N, pub_loader=None):
    results = []
    if args.mode == 'vanilla':
        for epoch in range(1, args.epochs + 1):
            train(model, train_loader, criterion, optimizer, device, epoch)
            accuracy = test(model, device, test_loader)
            results.append(accuracy)

    elif args.mode == 'gauss_sgd':
        for epoch in range(1, args.epochs + 1):
            train_gauss_sgd(args, model, train_loader, criterion, device)
            accuracy = test(model, device, test_loader)
            results.append(accuracy)

    elif args.mode == 'gauss_sgd_topk':
        for epoch in range(1, args.epochs + 1):
            train_gauss_sgd_topk(args, model, train_loader, criterion, device)
            accuracy = test(model, device, test_loader)
            results.append(accuracy)
    
    elif args.mode == 'public_aid_topk':
        for epoch in range(1, args.epochs + 1):
            train_public_aid_topk(args, model, train_loader, pub_loader, criterion, device)
            accuracy = test(model, device, test_loader)
            results.append(accuracy)
    
    if args.mode != 'vanilla':
        compute_dp_sgd_privacy(N, args.batch_size, args.noise_multiplier, args.epochs, args.delta)
    
    return results


def test(model, device, test_loader):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in tqdm(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )
    return correct / len(test_loader.dataset)

 
if __name__ == "__main__":

    # command line arguments for training. mainly hyperparameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.0015)
    parser.add_argument('--batch_size', type=int, default=60)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--momentum', type=float, default=0)
    # norm threshold for clipping gradients
    parser.add_argument('--max_grad_norm', type=float, default=0.9)
    
    parser.add_argument('--weights_path', type=str, default='./weights_no_compress')
    parser.add_argument('--acc_path', type=str, default='./acc_no_compress')
    parser.add_argument('--noise_multiplier', type=float, default=0.8)
    parser.add_argument('--topk_percentile', type=float, default=0.05)

    # decide training mode
    parser.add_argument('--mode', type=str, default='public_aid_topk')

    # delta value for privacy accounting
    parser.add_argument('--delta', type=float, default=1e-4)
    

    args = parser.parse_args()

     # prepare training & testing data
    transform = transforms.Compose(
    [
        transforms.ToTensor()
    ])

    trainset = torchvision.datasets.MNIST(root='./datasets', train=True,
                                            download=True, transform=transform)
                            
    testset = torchvision.datasets.MNIST(root='./datasets', train=False,
                                        download=True, transform=transform)
    
    for mode in ['public_aid_topk']:
        args.mode = mode
        if args.mode != 'public_aid_topk':
            train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                                shuffle=True)
            pub_loader = None
            N = len(trainset)
        else:
            train_subset, pub_subset = torch.utils.data.random_split(
            trainset, [59400, 600], generator=torch.Generator().manual_seed(1))

            train_loader = torch.utils.data.DataLoader(train_subset, batch_size=args.batch_size,
                                                shuffle=True)
            pub_loader = torch.utils.data.DataLoader(pub_subset, batch_size=args.batch_size, 
                                                shuffle=True)
            N = len(train_subset)
        
        test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                            shuffle=False)

        classes = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)


        USE_CUDA = torch.cuda.is_available()
        device = torch.device("cuda" if USE_CUDA else "cpu")
        print(f'training device: {device}')    
        print(f'training set size: {N}')


        net = models.SampleConvNet()
        net.to(device)
        criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        optimizer = optim.SGD(params=net.parameters(), lr=args.lr, momentum=args.momentum)
        args.mode = mode
        results = experiment(args, net, train_loader, criterion, optimizer, device, test_loader, N, pub_loader)
        print(f'mode: {args.mode}, result: {results}')
