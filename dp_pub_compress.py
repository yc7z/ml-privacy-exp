import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import argparse
import models
import matplotlib.pyplot as plt
from utils_plus import *
import numpy as np
from functorch import vmap, grad, make_functional
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


def train_vanilla_topk(args, model, train_loader, criterion, optimizer, device):
    model.train()
    losses = []
    for i, (data, target) in enumerate(tqdm(train_loader)):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        topk_compress(model, args.topk_percentile)
        optimizer.step()
        losses.append(loss.item())


def train_gauss_sgd(args, model, train_loader, criterion, optimizer, device):
    """
    Train model in a differentially private manner by clipping each per-sample gradient and adding noises.
    """

    model.train()
    for i, data in enumerate(tqdm(train_loader)):
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


def train_gauss_sgd_topk(args, model, train_loader, criterion, optimizer, device):
    """
    Train model in a differentially private manner by clipping each per-sample gradient and adding noises.

    Additionally, perform topk compression on gradients before each GD.
    """

    model.train()
    for i, data in enumerate(tqdm(train_loader)):
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

        # 4. gradient compression.
        topk_compress(model, args.topk_percentile)

        optimizer.step()
        optimizer.zero_grad()


def train_leaky_gauss_sgd(args, model, train_loader, pub_loader, criterion, optimizer, device):
    """
    Train model in a differentially private manner by clipping each per-sample gradient and adding noises.
    Compute gradient on public data and use its topk mask to compress privatized gradient.
    """
    model.train()
    for data, pub_data in zip(tqdm(train_loader), cycle(pub_loader)):
        inputs, targets = data
        inputs, targets = inputs.to(device), targets.to(device)
        
        pub_inputs, pub_targets = pub_data
        pub_inputs, pub_targets = pub_inputs.to(device), pub_targets.to(device)

        # 0. compute gradient on public data without updating it.
        pub_loss = criterion(model(pub_inputs), pub_targets)
        pub_grads = torch.autograd.grad(pub_loss, model.parameters())
        pub_masks = topk_mask_all(pub_grads, args.topk_percentile)
        model.zero_grad()

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

        # 4. perform topk compression but use the mask obtained from public gradient.
        apply_external_mask(model, pub_masks)

        optimizer.step()
        optimizer.zero_grad()


def train_leaky_gauss_sgd_topk(args, model, train_loader, criterion, optimizer, device):
    """
    Train model in a differentially private manner by clipping each per-sample gradient and adding noises.
    Compute gradient on public data and use its topk mask to compress privatized gradient.
    """
    model.train()
    for data in tqdm(train_loader):
        inputs, targets = data
        inputs, targets = inputs.to(device), targets.to(device)

        # 0. compute gradient on public data without updating it.
        pub_loss = criterion(model(inputs), targets)
        pub_grads = torch.autograd.grad(pub_loss, model.parameters())
        pub_masks = topk_mask_all(pub_grads, args.topk_percentile)
        model.zero_grad()

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

        # 4. perform topk compression but use the mask obtained from public gradient.
        apply_external_mask(model, pub_masks)

        optimizer.step()
        optimizer.zero_grad()


def experiment(args, model, train_loader, criterion, optimizer, device, test_loader, N, pub_loader=None):
    results = []
    if args.mode == 'vanilla':
        for epoch in range(1, args.epochs + 1):
            train(model, train_loader, criterion, optimizer, device, epoch)
            accuracy = test(model, device, test_loader)
            results.append(accuracy)

    elif args.mode == 'gauss_sgd':
        for epoch in range(1, args.epochs + 1):
            train_gauss_sgd(args, model, train_loader, criterion, optimizer, device)
            accuracy = test(model, device, test_loader)
            results.append(accuracy)

    elif args.mode == 'gauss_sgd_topk':
        for epoch in range(1, args.epochs + 1):
            train_gauss_sgd_topk(args, model, train_loader, criterion, optimizer, device)
            accuracy = test(model, device, test_loader)
            results.append(accuracy)
    
    elif args.mode == 'public_aid_topk':
        for epoch in range(1, args.epochs + 1):
            train_leaky_gauss_sgd(args, model, train_loader, pub_loader, criterion, optimizer, device)
            accuracy = test(model, device, test_loader)
            results.append(accuracy)
    
    elif args.mode == 'vanilla_topk':
        for epoch in range(1, args.epochs + 1):
            train_vanilla_topk(args, model, train_loader, criterion, optimizer, device)
            accuracy = test(model, device, test_loader)
            results.append(accuracy)
    
    elif args.mode == 'leaky_gauss_sgd_topk':
        for epoch in range(1, args.epochs + 1):
            train_leaky_gauss_sgd_topk(args, model, train_loader, criterion, optimizer, device)
            accuracy = test(model, device, test_loader)
            results.append(accuracy)
    
    if args.mode != 'vanilla' or args.mode != 'vanilla_topk' or args.mode != 'leaky_gauss_sgd_topk':
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
    parser.add_argument('--momentum', type=float, default=0.9)
    # norm threshold for clipping gradients
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    # control gaussian noise level
    parser.add_argument('--noise_multiplier', type=float, default=4.0)
    parser.add_argument('--topk_percentile', type=float, default=0.01)

    # decide training mode
    parser.add_argument('--mode', type=str, default='public_aid_topk')

    # delta value for privacy accounting
    parser.add_argument('--delta', type=float, default=1e-5)
    

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
    
    epoch_lst = [_ for _ in range(1, args.epochs + 1)]
    # ['public_aid_topk', 'gauss_sgd_topk', 'gauss_sgd']
    for mode in ['leaky_gauss_sgd_topk', 'gauss_sgd_topk', 'public_aid_topk', 'gauss_sgd', 'vanilla_topk']:
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
        # optimizer = optim.SGD(params=net.parameters(), lr=args.lr, momentum=args.momentum)
        optimizer = optim.Adam(params=net.parameters(), lr=args.lr)
        args.mode = mode
        results = experiment(args, net, train_loader, criterion, optimizer, device, test_loader, N, pub_loader)
        print(f'mode: {args.mode}, result: {results}')
        plt.plot(epoch_lst, results, label=mode)
    plt.legend(loc="lower right")
    plt.savefig('./accuracies_curves_Adam_1_percent(1).png')
