import torch
import torch.nn as nn

def fgsm_(model, x, target, eps, targeted=True, device='cpu', clip_min=None, clip_max=None):
    """Internal process for all FGSM and PGD attacks."""    
    # create a copy of the input, remove all previous associations to the compute graph...
    input_ = x.clone().detach_()
    # ... and make sure we are differentiating toward that variable
    input_.requires_grad_()

    # run the model and obtain the loss
    logits = model(input_)
    target = torch.LongTensor([target]).to(device)
    model.zero_grad()
    loss = nn.CrossEntropyLoss()(logits, target)
    loss.backward()
    
    # perfrom either targeted or untargeted attack
    if targeted:
        out = input_ - eps * input_.grad.sign()
    else:
        out = input_ + eps * input_.grad.sign()
    
    # if desired clip the ouput back to the image domain
    if (clip_min is not None) or (clip_max is not None):
        out.clamp_(min=clip_min, max=clip_max)
    return out

def pgd_(model, x, target, k, eps, eps_step, targeted=True, device='cpu', clip_min=None, clip_max=None):
    x_min = x - eps
    x_max = x + eps
    x_min = x_min.to(device)
    x_max = x_max.to(device)
    # generate a random point in the +-eps box around x
    x = torch.rand_like(x, device=device)
    x = (x*2*eps - eps)
    
    for i in range(k):
        # FGSM step
        x = fgsm_(model, x, target, eps_step, targeted, device)
        # Projection Step
        x = torch.max(x_min, x)
        x = torch.min(x_max, x)
    #if desired clip the ouput back to the image domain
    if (clip_min is not None) or (clip_max is not None):
        x.clamp_(min=clip_min, max=clip_max)
    return x

def pgd_untargeted(model, x, label, k, eps, eps_step, device='cpu', clip_min=None, clip_max=None, **kwargs):
    return pgd_(model, x, label, k, eps, eps_step, targeted=False, device=device, clip_min=clip_min, clip_max=clip_max, **kwargs)

def pgd_untargeted_batched(model, x_batch, y_batch, k, eps, eps_step, device='cpu', clip_min=None, clip_max=None, **kwargs):
    #################################################
    # Batched version of pgd_untargeted #
    #################################################
    x_batch_adv = []
    for x, y in zip(x_batch, y_batch):
        x_batch_adv.append(pgd_untargeted(model, x, y, k, eps, eps_step, device, clip_min, clip_max, **kwargs))

    return torch.stack(x_batch_adv)