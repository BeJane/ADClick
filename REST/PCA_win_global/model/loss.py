import torch
import torch.nn.functional as F

def focal_loss(inputs, targets, alpha: float=0.75, gamma: float=4, reduction: str = 'mean',loss_masks=None,beta=False):
    if targets.dim() != inputs.dim():
        targets = targets.unsqueeze(1)
        targets = torch.cat([torch.ones_like(targets)-targets,targets],dim=1)
    inputs = torch.softmax(inputs, dim=-1)

    y = torch.argmax(targets,dim=-1)#.unsqueeze(1)
    # pt = inputs.gather(1,y)
    pt = (inputs * targets).sum(1)
    loss = - (alpha * y +  (1-alpha) * (1-y)) * (1- pt)**gamma * pt.log()

    if beta:

        loss = loss*torch.max(targets,dim=-1)[0]
    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss
def focal_loss_mixmatch(inputs, targets, alpha: float=0.75, gamma: float=4, reduction: str = 'mean',loss_masks=None,beta=False):
    if targets.dim() != inputs.dim():
        targets = targets.unsqueeze(1)
        targets = torch.cat([torch.ones_like(targets)-targets,targets],dim=1)
    inputs = torch.softmax(inputs, dim=-1)

    y = torch.argmax(targets,dim=-1)#.unsqueeze(1)
    # follow mixmatch 有标签数据的cross entropy loss
    loss = - (alpha * y +  (1-alpha) * (1-y)) * ((1- inputs)**gamma * inputs.log() * targets).sum(1)

    # if beta:
    #     loss = loss*targets.gather(1,y)
    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss
def l2_loss(inputs, targets, alpha: float=0.75, reduction: str = 'mean'):
    if targets.dim() != inputs.dim():
        targets = targets.unsqueeze(1)
        targets = torch.cat([torch.ones_like(targets)-targets,targets],dim=1)
    inputs = torch.softmax(inputs, dim=-1)

    y = torch.argmax(targets,dim=-1)#.unsqueeze(1)
    # follow mixmatch 无标签数据的L2 loss
    loss = (alpha * y +  (1-alpha) * (1-y)) * ((inputs - targets)**2).mean(1)

    # if beta:
    #     loss = loss*targets.gather(1,y)
    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss

def semi_loss(logits,masks,label_kinds,iteration,args):
    label_kinds = label_kinds[:, 0]
    if args.semi_loss == 'focal_loss_mean':
        loss_x = focal_loss(logits[label_kinds == 1], masks[label_kinds == 1], alpha=args.focal_loss_alpha,
                            gamma=args.focal_loss_gamma, beta=args.focal_loss_beta,reduction='sum')if (label_kinds == 1).sum() > 0 else 0
        loss_u = focal_loss(logits[label_kinds == 0], masks[label_kinds == 0], alpha=args.u_focal_loss_alpha,
                            gamma=args.u_focal_loss_gamma, beta=args.focal_loss_beta,reduction='sum') if (label_kinds == 0).sum() > 0 else 0
        w = args.lambda_u * iteration / 400
        return (loss_x +loss_u * w)/label_kinds.flatten().shape[0]
    if args.semi_loss == 'focal_loss':
        loss_x = focal_loss(logits[label_kinds == 1], masks[label_kinds == 1], alpha=args.focal_loss_alpha,
                            gamma=args.focal_loss_gamma, beta=args.focal_loss_beta)if (label_kinds == 1).sum() > 0 else 0
        loss_u = focal_loss(logits[label_kinds == 0], masks[label_kinds == 0], alpha=args.u_focal_loss_alpha,
                            gamma=args.u_focal_loss_gamma, beta=args.focal_loss_beta) if (label_kinds == 0).sum() > 0 else 0
    if args.semi_loss == 'mixmatch':
        loss_x = focal_loss_mixmatch(logits[label_kinds == 1], masks[label_kinds == 1], alpha=args.focal_loss_alpha,
                            gamma=args.focal_loss_gamma, beta=args.focal_loss_beta) if (label_kinds == 1).sum() > 0 else 0
        loss_u = l2_loss(logits[label_kinds == 0], masks[label_kinds == 0], alpha=args.u_focal_loss_alpha) if (
                                                                                                   label_kinds == 0).sum() > 0 else 0
    if args.semi_loss == 'mixmatch_mean':
        loss_x = focal_loss_mixmatch(logits[label_kinds == 1], masks[label_kinds == 1], alpha=args.focal_loss_alpha,
                            gamma=args.focal_loss_gamma, beta=args.focal_loss_beta) if (label_kinds == 1).sum() > 0 else 0
        loss_u = l2_loss(logits[label_kinds == 0], masks[label_kinds == 0], alpha=args.u_focal_loss_alpha) if (
                                                                                                   label_kinds == 0).sum() > 0 else 0
        w = args.lambda_u * iteration / 400
        return (loss_x +loss_u * w)/label_kinds.flatten().shape[0]
    if args.semi_loss == 'focal_l2':
        loss_x = focal_loss(logits[label_kinds == 1], masks[label_kinds == 1], alpha=args.focal_loss_alpha,
                            gamma=args.focal_loss_gamma, beta=args.focal_loss_beta)if (label_kinds == 1).sum() > 0 else 0
        loss_u = l2_loss(logits[label_kinds == 0], masks[label_kinds == 0], alpha=args.u_focal_loss_alpha) if (
                                                                                                   label_kinds == 0).sum() > 0 else 0

    if args.semi_loss == 'mfocal_focal':
        loss_x = focal_loss_mixmatch(logits[label_kinds == 1], masks[label_kinds == 1], alpha=args.focal_loss_alpha,
                            gamma=args.focal_loss_gamma, beta=args.focal_loss_beta) if (label_kinds == 1).sum() > 0 else 0
        loss_u = focal_loss(logits[label_kinds == 0], masks[label_kinds == 0], alpha=args.u_focal_loss_alpha,
                            gamma=args.u_focal_loss_gamma, beta=args.focal_loss_beta) if (
                                                                                                     label_kinds == 0).sum() > 0 else 0
    # print(loss_x.item(),loss_u.item())
    w = args.lambda_u * iteration / 400
    # w = args.lambda_u * min(iteration / 400,1)
    return loss_x + w * loss_u
def consistency_loss(logits, targets, name='ce', mask=None,args=None):
    """
    consistency regularization loss in semi-supervised learning.

    Args:
        logits: logit to calculate the loss on and back-propagation, usually being the strong-augmented unlabeled samples
        targets: pseudo-labels (either hard label or soft label)
        name: use cross-entropy ('ce') or mean-squared-error ('mse') to calculate loss
        mask: masks to mask-out samples when calculating the loss, usually being used as confidence-masking-out
    """

    # assert name in ['ce', 'mse']
    # logits_w = logits_w.detach()
    if name == 'mse':
        probs = torch.softmax(logits, dim=-1)
        loss = F.mse_loss(probs, targets, reduction='none').mean(dim=1)
    if name == 'ce':
        loss = ce_loss(logits, targets, reduction='none')
    if name == 'focal_loss':
        loss = focal_loss(logits, targets, alpha=args.u_focal_loss_alpha,
                            gamma=args.u_focal_loss_gamma, beta=args.focal_loss_beta, reduction='none')
    if mask is not None:
        # mask must not be boolean type

        loss = loss * mask

    return loss.mean()

def  ce_loss(logits, targets, reduction='none'):
    """
    cross entropy loss in pytorch.

    Args:
        logits: logit values, shape=[Batch size, # of classes]
        targets: integer or vector, shape=[Batch size] or [Batch size, # of classes]
        # use_hard_labels: If True, targets have [Batch size] shape with int values. If False, the target is vector (default True)
        reduction: the reduction argument
    """
    if logits.shape == targets.shape:
        # one-hot target
        log_pred = F.log_softmax(logits, dim=-1)
        nll_loss = torch.sum(-targets * log_pred, dim=1)
        if reduction == 'none':
            return nll_loss
        else:
            return nll_loss.mean()
    else:
        log_pred = F.log_softmax(logits, dim=-1)
        return F.nll_loss(log_pred, targets, reduction=reduction)
