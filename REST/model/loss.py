import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from model import misc


class NormalizedFocalLossSigmoid(nn.Module):
    def __init__(self, axis=-1, alpha=0.25, gamma=2, max_mult=-1, eps=1e-12,
                 from_sigmoid=False, detach_delimeter=True,
                 batch_axis=0, weight=None, size_average=True,
                 ignore_label=-1):
        super(NormalizedFocalLossSigmoid, self).__init__()
        self._axis = axis
        self._alpha = alpha
        self._gamma = gamma
        self._ignore_label = ignore_label
        self._weight = weight if weight is not None else 1.0
        self._batch_axis = batch_axis

        self._from_logits = from_sigmoid
        self._eps = eps
        self._size_average = size_average
        self._detach_delimeter = detach_delimeter
        self._max_mult = max_mult
        self._k_sum = 0
        self._m_max = 0

    def forward(self, pred, label):
        one_hot = label > 0.5
        sample_weight = label != self._ignore_label

        if not self._from_logits:
            pred = torch.sigmoid(pred)

        alpha = torch.where(one_hot, self._alpha * sample_weight, (1 - self._alpha) * sample_weight)
        pt = torch.where(sample_weight, 1.0 - torch.abs(label - pred), torch.ones_like(pred))

        beta = (1 - pt) ** self._gamma

        sw_sum = torch.sum(sample_weight, dim=(-2, -1), keepdim=True)
        beta_sum = torch.sum(beta, dim=(-2, -1), keepdim=True)
        mult = sw_sum / (beta_sum + self._eps)
        if self._detach_delimeter:
            mult = mult.detach()
        beta = beta * mult
        if self._max_mult > 0:
            beta = torch.clamp_max(beta, self._max_mult)

        with torch.no_grad():
            ignore_area = torch.sum(label == self._ignore_label, dim=tuple(range(1, label.dim()))).cpu().numpy()
            sample_mult = torch.mean(mult, dim=tuple(range(1, mult.dim()))).cpu().numpy()
            if np.any(ignore_area == 0):
                self._k_sum = 0.9 * self._k_sum + 0.1 * sample_mult[ignore_area == 0].mean()

                beta_pmax, _ = torch.flatten(beta, start_dim=1).max(dim=1)
                beta_pmax = beta_pmax.mean().item()
                self._m_max = 0.8 * self._m_max + 0.2 * beta_pmax

        loss = -alpha * beta * torch.log(torch.min(pt + self._eps, torch.ones(1, dtype=torch.float).to(pt.device)))
        loss = self._weight * (loss * sample_weight)

        if self._size_average:
            bsum = torch.sum(sample_weight, dim=misc.get_dims_with_exclusion(sample_weight.dim(), self._batch_axis))
            loss = torch.sum(loss, dim=misc.get_dims_with_exclusion(loss.dim(), self._batch_axis)) / (bsum + self._eps)
        else:
            loss = torch.sum(loss, dim=misc.get_dims_with_exclusion(loss.dim(), self._batch_axis))

        return loss

    def log_states(self, sw, name, global_step):
        sw.add_scalar(tag=name + '_k', value=self._k_sum, global_step=global_step)
        sw.add_scalar(tag=name + '_m', value=self._m_max, global_step=global_step)


def sigmoid_adaptive_focal_loss(inputs, targets, num_masks=1, epsilon: float = 0.5, gamma: float = 2,
                                delta: float = 0.4, alpha: float = 1.0, eps: float = 1e-12):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        epsilon: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        delta: A Factor in range (0,1) to estimate the gap between the term of ∇B
                and the gradient term of bce loss.
        alpha: A coefficient of poly loss.
        eps: Term added to the denominator to improve numerical stability.
    Returns:
        Loss tensor
    """

    if targets.dim() != inputs.dim():
        targets = targets.unsqueeze(1)
        targets = torch.cat([torch.ones_like(targets)-targets,targets],dim=1)

    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)

    one_hot = targets > 0.5
    with torch.no_grad():
        p_sum = torch.sum(torch.where(one_hot, p_t, 0), dim=-1, keepdim=True)
        ps_sum = torch.sum(torch.where(one_hot, 1, 0), dim=-1, keepdim=True)
        gamma = gamma + (1 - (p_sum / (ps_sum + eps)))

    beta = (1 - p_t) ** gamma

    with torch.no_grad():
        sw_sum = torch.sum(torch.ones(p_t.shape, device=p_t.device), dim=-1, keepdim=True)
        beta_sum = (1 + delta * gamma) * torch.sum(beta, dim=-1, keepdim=True) + eps
        mult = sw_sum / beta_sum

    loss = mult * ce_loss * beta + alpha * (1 - p_t) ** (gamma + 1)

    if epsilon >= 0:
        epsilon_t = epsilon * targets + (1 - epsilon) * (1 - targets)
        loss = epsilon_t * loss
    # print(loss.mean(1).mean())
    return torch.mean(loss.mean(1))

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
