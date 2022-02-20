# Tools to deal with masking pytorch tensors with zeros corresponding to masked elements and ones to unmasked
from abc import ABC

from torch.nn import Module, Softmax
from torch.linalg import vector_norm


mask_values = {'masked': 0., 'unmasked': 1.}


def union_mask(mask1, mask2):
    if mask1 is None:
        return mask2
    elif mask2 is None:
        return mask1
    else:
        return mask1.minimum(mask2)


def masked_average2D(input, mask, mean_channels=False, keepdim=False, eps=1e-8):
    """
    Evaluates mean through the last 3 axes (except the last one 'if mean_channels' id False)
    Accounts only unmasked values (where mask value is 1.), returns 0. for all values masked
    Also evaluates new mask with zeros for batches where all elements were masked, otherwise ones

    :param input: torch tensor with 3 or more dimensions
    :param mask: None or mask tensor with 0. and 1., one dimension fewer than input
    :param mean_channels: bool, whether or not to calculate mean through the last axis
    :param keepdim: True to keep number of dimensions
    :param eps: addition to divisor to avoid NaN
    :return: averaged input, new mask
    """
    if mask is None:
        return input.mean((-3, -2, -1) if mean_channels else (-3, -2), keepdim), None
    else:
        n_unmasked = mask.sum((-2, -1), keepdim)  # Numbers of unmasked elements in batch
        if mean_channels:
            x = input.mean(-1)
            x *= mask
            average = x.sum((-2, -1), keepdim).div_(n_unmasked + eps)
            average = average.unsqueeze(-1) if keepdim else average
        else:
            x = input * mask.unsqueeze(-1)
            average = x.sum((-3, -2), keepdim).div_(n_unmasked.unsqueeze(-1) + eps)
        return average, n_unmasked.sign_()


class MaskedSoftmax(Softmax):
    """
    Softmax activation with mask
    """
    @staticmethod
    def backward_hook(m, grad_inputs, grad_outputs):
        # Replaces NaNs in gradient with zeros
        return tuple(grad.nan_to_num() for grad in grad_inputs)

    def __init__(self, dim=None):
        super().__init__(dim)
        self.register_full_backward_hook(MaskedSoftmax.backward_hook)

    def forward(self, input, mask=None):
        if mask is None:
            return super().forward(input)
        else:
            return super().forward(input - mask.reciprocal()).nan_to_num()


class MaskedNorm(Module):
    def __init__(self, dim=-1, eps=1.):
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self,input, mask=None):
        if mask is not None:
            input = input * mask.sign().add_(1.)
        norm = vector_norm(input, dim=self.dim, keepdim=True)
        return input / (norm + self.eps)
