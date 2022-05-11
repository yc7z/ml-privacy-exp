from statistics import mean
import torch

def compute_single_grad(model, loss_fn, sample, target):
    """
    Compute gradient of modle's loss w.r.t. a single training sample.
    """
    sample = sample.unsqueeze(0)
    target = target.unsqueeze(0)

    pred = model(sample)
    loss = loss_fn(pred, target)

    return torch.autograd.grad(loss, list(model.parameters()))


def compute_sample_grads(model, loss_fn, batch, targets):
    """
    Naively compute gradient for each sample within a batch by repeatedly calling compute_single_grad.
    """
    sample_grads = [compute_single_grad(model, loss_fn, batch[i], targets[i]) for i in range(batch.size(0))]
    sample_grads = zip(*sample_grads)
    sample_grads = [torch.stack(shards) for shards in sample_grads]
    return sample_grads


def compute_loss_stateless_model (fmodel, loss_fn, params, buffers, sample, target):
    """
        Compute model loss, but use a stateless functional version of model (fmodel) on a single sample and target.
    """
    batch = sample.unsqueeze(0)
    targets = target.unsqueeze(0)

    predictions = fmodel(params, buffers, batch) 
    loss = loss_fn(predictions, targets)
    return loss


def batch_clip(grads, max_norm):
    """
    Perform clipping given the gradient per each sample and clipping threshold max_norm.

    grads: a tuple of gradients, where each element is the gradients of one model parameter w.r.t. all samples within a minibatch.

    max_norm: the threshold for clipping.
    """
    batch_size = grads[0].size(0)
    grad_norms = []
    for grad_p in grads:
        grad_p_flat = grad_p.view(batch_size, -1)
        grad_norms.append(torch.norm(grad_p_flat, dim=1))
    grad_norms = torch.stack(grad_norms, dim=1)
    ones = torch.ones(size=grad_norms.size())
    scale_factors = torch.maximum(grad_norms / max_norm, ones)
    
    clipped_grads = [ torch.einsum("i...,i->i...", grads[k], scale_factors[:,k]) for k in range(len(grads)) ]

    return clipped_grads

def batch_noising(grads, clip, stddev=1.0, noise_multiplier=0.3):
    for grad_p in grads:
        grad_p += noise_multiplier * clip * stddev * torch.randn(size=grad_p.size())
    return grads
