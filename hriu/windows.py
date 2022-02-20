# Class to create non-overlapping windows
from torch import full
from torch.nn.functional import pad

from .utils import tuple_n
from .mask_tools import mask_values

class SpanningWindows:

    def __init__(self, window_size, window_stride=None, shift=0,
                 pad_value=.0, pad_mask=mask_values['masked'], unmasked=mask_values['unmasked']):
        """

        :param window_size: integer or 2-tuple
        :param window_stride: integer or 2-tuple, equal to 'window_size' by default
        :param shift:
        :param pad_value:
        :param pad_mask:
        :param unmasked:
        """

        self.window_size = tuple_n(2, window_size, 'window_size')

        if window_stride is None:
            self.window_stride = self.window_size
        else:
            self.window_stride = tuple_n(2, window_stride, 'window_stride')

        self.pad_value = pad_value
        self.pad_mask = pad_mask
        self.unmasked = unmasked

        if isinstance(shift, str):
            if shift.lower() == 'center':
                self.shift = None
                self.back_offset = (0, 0)
            else:
                raise ValueError("'shift' must be 2-tuple, integer or 'center'")
        else:
            self.shift = tuple_n(2, shift, 'shift')
            self.back_offset = tuple(
                (-x) % y for x, y in zip(self.shift, self.window_stride)
            )

    @staticmethod
    def num_wins(length, back_offset, window_size, window_stride):
        base = window_size - back_offset - length
        # Must be negative to get desirable reminder
        if base < 0:
            neg_n, residual = divmod(base, window_stride)
            return 1 - neg_n, residual
        else:
            return 1, base

    def __call__(self, x, mask=None):
        original_size = x.shape

        assert mask is None or original_size[:-1] == mask.shape

        win_in_col, add_to_col = SpanningWindows.num_wins(original_size[-3], self.back_offset[0], self.window_size[0],
                                                          self.window_stride[0])
        win_in_row, add_to_row = SpanningWindows.num_wins(original_size[-2], self.back_offset[1], self.window_size[1],
                                                          self.window_stride[1])

        if self.shift is not None:
            # Padding in Pytorch starts from the last axis
            padding = (self.back_offset[1], add_to_row,
                       self.back_offset[0], add_to_col)
        else:
            padding = (add_to_row // 2, add_to_row - add_to_row // 2,
                       add_to_col // 2, add_to_col - add_to_col // 2)

        if any(padding):
            x = pad(x, (0, 0) + padding, value=self.pad_value)
            if mask is not None:
                mask = pad(mask, padding, value=self.pad_mask)
            else:
                mask = full(x.shape[:-1], fill_value=self.pad_mask, dtype=x.dtype, device=x.device)
                mask[..., padding[2]:mask.shape[-2]-padding[3], padding[0]:mask.shape[-1]-padding[1]] = self.unmasked

        new_size = original_size[:-3] + (win_in_col, win_in_row) + self.window_size + original_size[-1:]

        original_stride = x.stride()
        new_stride = (original_stride[:-3] +
                      (original_stride[-3] * self.window_stride[0],
                       original_stride[-2] * self.window_stride[1]) +
                      original_stride[-3:])

        output = x.as_strided(new_size, new_stride, x.storage_offset())

        if mask is not None:
            mask_stride = mask.stride()
            new_mask_stride = (mask_stride[:-2] +
                               (mask_stride[-2] * self.window_stride[0],
                                mask_stride[-1] * self.window_stride[1]) +
                               mask_stride[-2:])
            new_mask = mask.as_strided(new_size[:-1], new_mask_stride, mask.storage_offset())
            return output, new_mask, padding
        else:
            return output, None, padding


def join_windows(x, mask=None, padding=None):
    original_size = x.shape
    new_size = (original_size[:-5] +
                (original_size[-5] * original_size[-3],
                 original_size[-4] * original_size[-2]) +
                original_size[-1:])
    output = x.transpose(-3, -4).reshape(new_size)
    if padding is not None:
        padding = (padding[0], output.shape[-2] - padding[1],
                   padding[2], output.shape[-3] - padding[3])
        output = output[..., padding[2]:padding[3], padding[0]:padding[1], :]
    if mask is not None:
        new_mask = mask.transpose(-2, -3).reshape(new_size[:-1])
        if padding is not None:
            new_mask = new_mask[..., padding[2]:padding[3], padding[0]:padding[1]]
        return output, new_mask
    else:
        return output, None
