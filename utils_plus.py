import torch


def batch_clip(model, max_norm):
    """
    Perform clipping given the gradient per each sample and clipping threshold max_norm.

    max_norm: the threshold for clipping.
    """
    grads = [param.grad_sample for param in model.parameters()]
    batch_size = grads[0].size(0)
    grad_norms = []
    for grad_p in grads:
        grad_p_flat = grad_p.view(batch_size, -1)
        grad_norms.append(torch.norm(grad_p_flat, dim=1))
    grad_norms = torch.stack(grad_norms, dim=1)
    ones = torch.ones(size=grad_norms.size(), device=grad_norms.device)
    scale_factors = torch.maximum(grad_norms / max_norm, ones)
    scale_factors = torch.reciprocal(scale_factors)
    
    for k, param in zip(range(len(grads)), model.parameters()):
        param.grad_sample = torch.einsum("i...,i", grads[k], scale_factors[:,k])


def batch_noising_scale(model, clip, noise_multiplier, batch_size):
    """
    Add to the gradient of each parameter a multivariate gaussian
    whose covariance matrix is 
    ``clip * noise_numtiplier * Identity.''
    """
    for param in model.parameters():
        param.grad = (param.grad_sample + torch.normal(0, clip * noise_multiplier, param.grad_sample.shape, device=param.grad_sample.device)) / batch_size
        del param.grad_sample


def topk_compress(model, percentile):
    """
    Compress the gradients of model parameters, keeping only the components whose magnitude is in the top percentile.
    """
    for parameter in model.parameters():
        grad_p_flat = parameter.grad.flatten()
        k = int(len(grad_p_flat) * percentile)
        topk_vals, topk_inds = torch.topk(input=torch.abs(grad_p_flat), k=k)
        mask = torch.zeros(size=grad_p_flat.shape).to(topk_inds.get_device())
        mask.scatter_(0, topk_inds, 1, reduce='add')
        parameter.grad = torch.multiply(mask, grad_p_flat).reshape(shape=parameter.grad.shape)


def topk_mask_single(grad_p, percentile):
    """
    return topk mask of grad_p based on percentile.
    """
    grad_p_flat = grad_p.flatten()
    k = int(len(grad_p_flat) * percentile)
    topk_vals, topk_inds = torch.topk(input=torch.abs(grad_p_flat), k=k)
    mask = torch.zeros(size=grad_p_flat.shape).to(topk_inds.get_device())
    mask.scatter_(0, topk_inds, 1, reduce='add')
    return mask.reshape(shape=grad_p.shape)


def topk_mask_all(grads, percentile):
    masks = []
    for grad_p in grads:
        masks.append(topk_mask_single(grad_p, percentile))
    return masks


def apply_external_mask(model, ext_masks):
    for parameter, mask in zip(model.parameters(), ext_masks):
        parameter.grad = torch.multiply(mask, parameter.grad)
