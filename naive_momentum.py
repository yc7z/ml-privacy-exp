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


def train_vanilla(args, model, trainloader, criterion, optimizer, device):
    """
        Train model without any privacy mechanisms.
    """
    print('start training: vanilla sgd (no momentum).')
    for epoch in range(args.epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            optimizer.zero_grad()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss
            if i % 50 == 0:    # save weights every 50 mini-batches
                torch.save(model.state_dict(), f'{args.weights_path}/vanilla/vanilla,epochs={epoch},batch={i}.pth')

        print(f'epoch {epoch + 1} finished.')
    print('finished vanilla training.')


def train_private(args, model, trainloader, criterion, device):
    """
    Train model in a differentially private manner by clipping each per-sample gradient and adding noises.
    """

    ft_compute_grad = grad(compute_loss_stateless_model, argnums=2)
    ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=(None, None, None, None, 0, 0))

    print('start training: private sgd.')
    for epoch in range(1, args.epochs + 1):
        for i, data in enumerate(trainloader):
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
            
                # 4. add gaussian noise
                batch_grads = batch_noising(batch_grads, clip=args.max_grad_norm, noise_multiplier=args.noise_multiplier)

                # 5. Update model parameters via gradient descent.
                for param, grad_p_b in zip(model.parameters(), batch_grads):
                    param -= args.lr * grad_p_b
            
            if i % 50 == 0:
                torch.save(model.state_dict(), f'{args.weights_path}/private_sgd/private_sgd,epochs={epoch},batch={i},clip={args.max_grad_norm},noise_mult={args.noise_multiplier}.pth')

        print(f'epoch {epoch} finished.')

    print('finished training.')


def train_private_naive_momentum(args, model, trainloader, criterion, device):
    """
    Train model in a differentially private manner by clipping each per-sample gradient and adding noises.
    """

    ft_compute_grad = grad(compute_loss_stateless_model, argnums=2)
    ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=(None, None, None, None, 0, 0))

    print('start training: naive momentum (private).')
    for epoch in range(1, args.epochs + 1):
        for i, data in enumerate(trainloader):
            inputs, targets = data
            inputs = inputs.to(device)
            targets = targets.to(device)

            # 1. Compute the gradient w.r.t. each model parameter on each sample within a batch.
            fmodel, params, buffers = make_functional_with_buffers(model)
            grads = ft_compute_sample_grad(fmodel, criterion, params, buffers, inputs, targets)

            # 2. clip each per-sample gradient
            grads = batch_clip(grads, args.max_grad_norm)

            # 3. take mean of grads over a batch
            batch_grads = []
            for grad_p in grads:
                batch_grads.append(torch.mean(grad_p, dim=0))
            
            # 4. add gaussian noise
            batch_grads = batch_noising(batch_grads, clip=args.max_grad_norm, noise_multiplier=args.noise_multiplier)

            # 5. (re)initialize the gradient momentum periodically
            if (i + 1) % args.accum_period == 1:
                grad_accumulation = init_accumulation(batch_grads)
            
            for j in range(len(batch_grads)):
                    grad_accumulation[j] += batch_grads[j].data

            # 6. Update model parameters via gradient descent as usual.
            with torch.no_grad():
                for param, grad_p_b in zip(model.parameters(), batch_grads):
                    param -= args.lr * grad_p_b
            
                # 7. when gradient accumulation period is hit, update model parameters with momentum.
                if i != 0 and i % args.accum_period == 0:
                    for param, grad_accum in zip(model.parameters(), grad_accumulation):
                        param -= (args.lr / args.accum_period) * grad_accum
            
            if i % 50 == 0:
                torch.save(model.state_dict(), f'{args.weights_path}/private_naive_momentum/naive_momentum,epochs={epoch},batch={i},clip={args.max_grad_norm},noise_mult={args.noise_multiplier},accumu_period={args.accum_period}.pth')

        print(f'epoch {epoch} finished.')
    print('finished training.')


def train_nonperiodic_momentum(args, model, trainloader, criterion, device):
    """
    Train model in a differentially private manner by clipping each per-sample gradient and adding noises.
    """

    ft_compute_grad = grad(compute_loss_stateless_model, argnums=2)
    ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=(None, None, None, None, 0, 0))

    print('start training: nonperiodic momentum (private).')
    for epoch in range(1, args.epochs + 1):
        for i, data in enumerate(trainloader):
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
                
                # 4. add gaussian noise
                batch_grads = batch_noising(batch_grads, clip=args.max_grad_norm, noise_multiplier=args.noise_multiplier)

                # 5. initialize the gradient momentum.
                if i == 0:
                    grad_accumulation = init_accumulation(batch_grads)
                    num_accum = 0

                for j in range(len(batch_grads)):
                    grad_accumulation[j] = (grad_accumulation[j] * num_accum + batch_grads[j].data) / (num_accum + 1)
                num_accum += 1

                # 6. Update model parameters via gradient descent as usual.
                for param, grad_p_b, grad_accum in zip(model.parameters(), batch_grads, grad_accumulation):
                    param -= args.lr * (0.5 * grad_p_b + (0.5 / num_accum) * grad_accum)
                    # param -= args.lr * (1 * grad_p_b + (0.0 / num_accum) * grad_accum)
                    # param -= args.lr * (0.5 * grad_p_b + 0.0025 * grad_accum)
                    # param -= args.lr * ((1 - 0.5 / num_accum) * grad_p_b + (0.5 / num_accum) * grad_accum)


            
            if i % 50 == 0:
                torch.save(model.state_dict(), f'{args.weights_path}/np_momentum/np_momentum,epochs={epoch},batch={i},clip={args.max_grad_norm},noise_mult={args.noise_multiplier}.pth')

            del grads
            del batch_grads

        print(f'epoch {epoch} finished.')
    print('finished training.')


def train_torch_momentum(args, model, trainloader, criterion, device):
    """
    Train model in a differentially private manner by clipping each per-sample gradient and adding noises.
    """

    ft_compute_grad = grad(compute_loss_stateless_model, argnums=2)
    ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=(None, None, None, None, 0, 0))

    print('start training: torch momentum (private).')
    for epoch in range(1, args.epochs + 1):
        for i, data in enumerate(trainloader):
            inputs, targets = data
            inputs = inputs.to(device)
            targets = targets.to(device)

            # 1. Compute the gradient w.r.t. each model parameter on each sample within a batch.
            fmodel, params, buffers = make_functional_with_buffers(model)
            grads = ft_compute_sample_grad(fmodel, criterion, params, buffers, inputs, targets)

            # 2. clip each per-sample gradient
            grads = batch_clip(grads, args.max_grad_norm)

            # 3. take mean of grads over a batch
            batch_grads = []
            for grad_p in grads:
                batch_grads.append(torch.mean(grad_p, dim=0))
            
            # 4. add gaussian noise
            batch_grads = batch_noising(batch_grads, clip=args.max_grad_norm, noise_multiplier=args.noise_multiplier)

            # 5. initialize the gradient momentum.
            if i == 0:
                grad_velocity = init_accumulation(batch_grads)

            for j in range(len(batch_grads)):
                grad_velocity[j] = grad_velocity[j] * args.momentum + batch_grads[j].data

            # 6. Update model parameters via gradient descent as usual.
            with torch.no_grad():
                for param, grad_v in zip(model.parameters(), grad_velocity):
                    param -= args.lr * grad_v
            
            if i % 50 == 0:
                torch.save(model.state_dict(), f'{args.weights_path}/torch_momentum/torch_momentum,epochs={epoch},batch={i},clip={args.max_grad_norm},noise_mult={args.noise_multiplier},momentum={args.momentum}.pth')

            del grads
            del batch_grads

        print(f'epoch {epoch} finished.')
    print('finished training.')


def eval_classifier(model, weights_path, testloader, device):
    """
        Load pretrained weights and evaluate model performance.
    """
    model.load_state_dict(torch.load(weights_path))
    total_correct = 0
    total = 0

    with torch.no_grad():
        for data in testloader:
            images, true_labels = data
            images = images.to(device)
            true_labels = true_labels.to(device)

            logits = model(images)
            predictions = torch.argmax(logits, dim=1)
            wrong_pred = torch.count_nonzero(predictions - true_labels)
            total += len(true_labels)
            total_correct += len(true_labels) - wrong_pred
    
    acc = total_correct / total
    print(f'accuracy on test set: {acc}')
    return acc


def get_test_accuracy(directory_str, model, testloader, device):
    """
        load all weights in directory_str and evaluate model accuracy following the order in which the weights are saved.
    """
    directory = os.fsencode(directory_str)
    accuracy_progress = []
    files = os.listdir(directory)
    files.sort(key=lambda x: os.stat(os.path.join(directory, x)).st_ctime)
    for file in files:
        filename = os.fsdecode(file)
        print(filename)
        weights_path = directory_str + '/' + filename
        accuracy_progress.append(eval_classifier(model, weights_path, testloader, device))
    return accuracy_progress
 
if __name__ == "__main__":

    # command line arguments for training. mainly hyperparameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=4e-3)
    parser.add_argument('--batch_size', type=int, default=120)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--momentum', type=float, default=0)
    # norm threshold for clipping gradients
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    
    parser.add_argument('--weights_path', type=str, default='./weights_no_compress')
    parser.add_argument('--acc_path', type=str, default='./acc_no_compress')
    parser.add_argument('--noise_multiplier', type=float, default=1.2)
    parser.add_argument('--topk_percentile', type=float, default=0.005)

    # accumulate gradient every accum_period minibatches.
    parser.add_argument('--accum_period', type=int, default=10)

    # arguments that determine experiments type
    parser.add_argument('--clip', type=bool, default=False)
    parser.add_argument('--noising', type=bool, default=False)
    parser.add_argument('--compress', type=bool, default=False)
    parser.add_argument('--accumulate_grad', type=bool, default=False)

    # whether we run training or evaluating trained model on test data.
    parser.add_argument('--eval', type=bool, default=False)

    # decide training mode
    parser.add_argument('--mode', type=str, default='private_sgd')

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
    print(f'training set size: {len(trainloader)}')
    
    
    net = models.LeNet(10, input_channel=1)
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')
    optimizer = optim.SGD(params=net.parameters(), lr=args.lr, momentum=args.momentum)

    if not args.eval:
        if args.mode == 'vanilla':
            train_vanilla(args, net, trainloader, criterion, optimizer, device)
        elif args.mode == 'private_sgd':
            train_private(args, net, trainloader, criterion, device)
        elif args.mode == 'private_naive_momentum':
            train_private_naive_momentum(args, net, trainloader, criterion, device)
        elif args.mode == 'np_momentum':
            train_nonperiodic_momentum(args, net, trainloader, criterion, device)
        elif args.mode == 'torch_momentum':
            train_torch_momentum(args, net, trainloader, criterion, device)
        
    else:

        weights_dir = {
            # 'vanilla': f'{args.weights_path}/vanilla',
            # 'private_sgd': f'{args.weights_path}/private_sgd',
            # 'private_naive_momentum': f'{args.weights_path}/private_naive_momentum'
            'np_momentum': f'{args.weights_path}/np_momentum',
            # 'torch_momentum': f'{args.weights_path}/torch_momentum'
        }

        for mode_name, target_path in weights_dir.items():
            model_acc = get_test_accuracy(target_path, net, testloader, device)
            with open(args.acc_path + '/' + mode_name, 'wb') as fp:
                pickle.dump(model_acc, fp)
                print(f'accuracy results saved to {args.acc_path}/{mode_name}')
            fp.close()