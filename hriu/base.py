# Hyperbolic Relative Imprinting Unit

import numpy as np

#import torch
from torch import empty, cat, einsum, full, full_like, empty_like, zeros_like, ones, std_mean
from torch import bool as torch_bool
from torch.nn import Module, Parameter, Linear
from torch.nn.init import uniform_

from utils import tuple_n, prod, iterate
from mask_tools import MaskedSoftmax, MaskedNorm, mask_values, masked_average2D, union_mask
from windows import SpanningWindows, join_windows


norm_dims = (-3, -2)
renorm = MaskedNorm


class TranslationSymmetric(Module):
    # Define parameters which is symmetric with respect to spatial shift
    def __init__(self, window_size, shapes, projective_size=None, num_patterns=1, join_xy=True):
        """
        Creates pairwise sets of parameters for each point in 2-d input window and 2-d projective window
        which is symmetric with respect to spatial shift.

        :param window_size: integer or 2-tuple
        :param shapes: tuple of parameter's shapes
        :param projective_size: integer or 2-tuple, equal 'window_size' by default
        :param num_patterns: integer, number of parameter's collections (e.g. number of heads for self-attention)
        :param join_xy: whether to join spatial dimensions into a single axis
        """
        super().__init__()
        self.window_size = tuple_n(2, window_size, 'window_size')
        self.shapes = shapes
        self.vector_sizes = tuple(prod(shape) for shape in self.shapes)
        self.num_patterns = num_patterns

        if projective_size is None:
            self.max_size = self.projective_size = self.window_size
            upscale = downscale = (1, 1)
        else:
            self.projective_size = tuple_n(2, projective_size, 'projective_size')
            self.max_size = tuple(max(x, y) for x, y in zip(self.window_size, self.projective_size))
            upscale = tuple(max(y // x, 1) for x, y in zip(self.window_size, self.projective_size))
            downscale = tuple(max(x // y, 1) for x, y in zip(self.window_size, self.projective_size))

        map_shape = self.projective_size + (self.num_patterns,) + self.window_size

        self.join_xy = join_xy
        if self.join_xy:
            map_shape1 = (map_shape[0] * map_shape[1],
                          self.num_patterns,
                          map_shape[3] * map_shape[4])
        else:
            map_shape1 = map_shape
        self.out_shapes = tuple(map_shape1 + tuple(iterate(shape)) for shape in shapes)

        self.intrinsic_size = (self.num_patterns,
                               self.max_size[0] * 2 - max(upscale[0], downscale[0]),
                               self.max_size[1] * 2 - max(upscale[1], downscale[1]),
                               sum(self.vecor_sizes))
        self.intrinsic = Parameter(empty(self.intrinsic_size))
        self.reset_parameters()

        i_stride = self.intrinsic.stride()
        temp_shape = map_shape + self.intrinsic_size[-1:]
        temp_stride = (
            i_stride[-3] * downscale[0],
            i_stride[-2] * downscale[1],
            i_stride[-4],
            i_stride[-3] * upscale[0],
            i_stride[-2] * upscale[1],
            i_stride[-1]
        )
        self.stride_args = (temp_shape, temp_stride)

    def reset_parameters(self):
        vars = self.intrinsic.split(self.vecor_sizes, -1)
        for var, size in zip(vars, self.vecor_sizes):
            bound = np.sqrt(6 / (prod(self.window_size) * size + prod(self.projective_size)))
            uniform_(var, -bound, bound)

    def get(self):
        vars = self.intrinsic.as_strided(*self.stride_args, self.intrinsic.storage_offset())
        vars = vars.flip((-3, -2))
        vars = vars.split(self.vecor_sizes, -1)
        return tuple(var.reshape(out_shape)
                     for var, out_shape in zip(vars, self.out_shapes))


class MLP(Module):
    def __init__(self, in_channels, out_channels=None, hidden=None, factor=None, activation=None):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.activation = activation
        if hidden or factor:
            self.hidden = hidden or self.in_channels * factor
        else:
            raise ValueError("One of 'hidden', 'factor' must be set")

        self.first = Linear(self.in_channels, self.hidden)
        self.last = Linear(self.hidden, self.out_channels)

    def forward(self, input):
        x = self.first(input)
        x = self.activation(x)
        x = self.last(x)
        x = self.activation(x)
        return x


class InstanceNorm(Module):
    def __init__(self, norm_dims, affine_shape, eps = 1e-8):
        super().__init__()
        self.norm_dims = tuple(iterate(norm_dims))
        self.eps = eps

        self.gamma = Parameter(ones(affine_shape))
        self.beta = Parameter(zeros_like(self.gamma))

    def forward(self, input):
        std, mean = std_mean(input, dim=self.norm_dims, unbiased=False, keepdim=True)
        x = input - mean
        x /= std + self.eps
        x *= self.gamma
        x += self.beta
        return x


class HRIU(Module):
    """
    Hyperbolic Relative Imprinting Unit
    """

    def __init__(self, window_size, channels, shift=0, projective_size=None,
                 num_heads=1, head_dim=None, output_channels=None,
                 mid_activation=None, activation=None,
                 technique=None,
                 mode=None,
                 scale=None, dropout=None):
        super().__init__()

        self.window_size = tuple_n(2, window_size, 'window_size')
        self.num_patches = self.window_size[0] * self.window_size[1]

        if projective_size is None:
            self.projective_size = self.window_size
            self.resize = False
        else:
            self.projective_size = tuple_n(2, projective_size, 'projective_size')
            self.resize = self.window_size != self.projective_size
        self.proj_patches = self.projective_size[0] * self.projective_size[1]

        self.partition = SpanningWindows(window_size=window_size, shift=shift)

        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = head_dim or channels // num_heads
        self.output_channels = output_channels or self.channels

        self.scale = scale or self.head_dim ** -0.5
        self.dropout = dropout

        if mode is None:
            self.return_keys = self.use_extra_keys = False
        else:
            mode = mode.lower()
            self.return_keys = mode == 'encoder'
            self.use_extra_keys = mode == 'decoder'

        if technique is None:
            if self.resize:
                self.attention = self.involution = False
                self.imprinting = True
            else:
                self.attention = self.involution = self.imprinting = True
        else:
            technique = technique.lower()
            if technique == 'all':
                self.attention = self.involution = self.imprinting = True,
            elif len(technique) <= 3:
                self.attention = 'a' in technique
                self.involution = 'n' in technique
                self.imprinting = 'm' in technique
            else:
                self.attention = 'atten' in technique
                self.involution = 'invol' in technique
                self.imprinting = 'impr' in technique
            if self.resize:
                if self.attention or self.involution:
                    print('WARNING: attention and involution disabled due to rescaling')
                self.attention = self.involution = False

        self.recalc_mask = not (self.attention or self.involution)

        self.num_queries = self.attention + self.involution
        self.num_keys = (1 + self.attention + self.imprinting)
        self.num_patterns = self.num_heads

        if self.use_extra_keys:
            self.num_queries *= 2
            self.num_patterns *= 2

        self.num_qk = self.num_queries + self.num_keys

        obtained_size = self.num_heads * self.head_dim * self.num_qk
        self.spatial_parameters_shapes = (((),) +
                                          (self.head_dim,) * (self.involution + self.imprinting))

        self.prepare = Linear(channels, obtained_size)
        self.mid_activation = mid_activation

        self.spatial = TranslationSymmetric(
            self.window_size, self.spatial_parameters_shapes,
            projective_size, num_patterns=self.num_patterns, join_xy=True)

        #self.renorm = MaskedSoftmax(dim=-1)
        self.renorm = MaskedNorm(dim=-1)

        self.finish = Linear(self.num_heads * self.head_dim, self.output_channels)
        self.activation = activation

    def forward(self, input, mask=None, extra_keys=None, extra_mask=None):
        """
            input.shape: [batch, H, W, C]
            mask.shape: [batch, H, W]
            extra_keys.shape: [num_keys, batch*Nwins, num_heads, winH*winW, head_dim]
            extra_mask.shape: [batch*Nwin, 1, 1, 1, winH*winW]
        """
        x = self.prepare(input)
        if self.mid_activation:
            x = self.mid_activation(x)
        # x.shape: [batch, H, W, num_keys*num_heads*head_dim]

        windows, parts_mask, padding = self.partition(x, mask)
        wins_shape = windows.shape
        b_wic_win = windows.shape[:3]
        b_num_windows = windows.shape[0] * windows.shape[1] * windows.shape[2]

        # windows.shape: [batch, win_in_cols, win_in_rows, winH, winW, num_qk*num_heads*head_dim]
        # parts_mask.shape: [batch, win_in_cols, win_in_rows, winH, winW]

        windows = windows.unflatten(-1, (self.num_qk, self.num_heads, self.head_dim))
        # windows.shape: [batch, win_in_cols, win_in_rows, winH, winW, num_qk, num_heads, head_dim]

        queries, keys = windows.split((self.num_queries, self.num_keys), dim=-3)
        # queries.shape: [batch, win_in_cols, win_in_rows, winH, winW, num_queries, num_heads, head_dim]
        # keys.shape: [batch, win_in_cols, win_in_rows, winH, winW, num_keys, num_heads, head_dim]

        queries = queries.view(wins_shape[:-1] + (-1, self.num_patterns, self.head_dim))
        # queries.shape: [batch, win_in_cols, win_in_rows, winH, winW, num_queries?, num_patterns, head_dim]

        queries = queries.moveaxis((-3, -2), (0, -4))
        # queries.shape: [num_queries?, batch, win_in_cols, win_in_rows, num_patterns, winH, winW, head_dim]

        keys = keys.moveaxis((-3, -2), (0, -4))
        # queries.shape: [num_keys, batch, win_in_cols, win_in_rows, num_heads, winH, winW, head_dim]

        if self.use_extra_keys:
            keys = cat((keys, extra_keys.view_as(keys)), dim=-4)

        # ***Physical copy***
        queries = queries.reshape((-1, b_num_windows, self.num_patterns, self.num_patches, self.head_dim))
        # queries.shape: [num_queries?, batch*Nwins, num_patterns, winH*winW, head_dim]
        keys = keys.reshape((self.num_keys, b_num_windows, self.num_patterns, self.num_patches, self.head_dim))
        # queries.shape: [num_keys, batch*Nwins, num_patterns, winH*winW, head_dim]

        queries_i = iter(queries.unbind(dim=0))
        keys_i = iter(keys.unbind(dim=0))

        spatial_parameters = iter(self.spatial.get())

        score = next(spatial_parameters)
        # score.shape: [outwinH*outwinW, num_patterns, winH*winW]

        score = score.view(self.proj_patches, self.num_heads, -1, self.num_patches)
        # score.shape: [outwinH*outwinW, num_heads, 1?2, winH*winW]

        # ***Physical copy***
        score = score.repeat(b_num_windows, 1, 1, 1, 1)
        # score.shape: [batch*Nwins, outwinH*outwinW, num_heads, 1?2, winH*winW]

        if self.involution:
            # Must be true: outwinH*outwinW == winH*winW
            q_n = next(queries_i)
            # q_n.shape: [batch*Nwins, num_patterns, winH*winW, head_dim]
            pattern_n = next(spatial_parameters)
            # pattern_n.shape: [outwinH*outwinW, num_patterns, winH*winW, head_dim]
            score += einsum('...hod,...ohid->...ohi', q_n, pattern_n).view_as(score)
            # score.shape: [batch*Nwins, outwinH*outwinW, num_heads, 1?2, winH*winW]

        if self.attention:
            # Must be true: outwinH*outwinW == winH*winW
            q = next(queries_i)
            # q.shape: [batch*Nwins, num_patterns, winH*winW, head_dim]
            k = next(keys_i)
            # k.shape: [batch*Nwins, num_patterns, winH*winW, head_dim]
            score += einsum('...hod,...hid->...ohi', q, k).view_as(score)
            # score.shape: [batch*Nwins, outwinH*outwinW, num_heads, 1?2, winH*winW]

        if self.imprinting:
            k_m = next(keys_i)
            # k_m.shape: [batch*Nwins, num_patterns, winH*winW, head_dim]
            pattern_m = next(spatial_parameters)
            # pattern_m.shape: [outwinH*outwinW, num_patterns, winH*winW, head_dim]
            score += einsum('...hid,ohid->...ohi', k_m, pattern_m).view_as(score)
            # score.shape: [batch*Nwins, outwinH*outwinW, num_heads, 1?2, winH*winW]

        ########        score *= self.scale

        is_extramask = (extra_mask is not None and self.use_extra_keys)
        is_drop = (self.dropout and self.training)

        if (parts_mask is None) and not is_extramask and not is_drop:
            used_mask = None
            score = score.flatten(-2)
            score = self.renorm(score)

        else:
            # parts_mask.shape: [batch, win_in_cols, win_in_rows, winH, winW]
            if parts_mask is None:
                parts_mask = full((b_num_windows, 1, 1, 1, self.num_patches),
                                  fill_value= mask_values['unmasked'],
                                  device=score.device,
                                  dtype=score.dtype)
            else:
                parts_mask = parts_mask.reshape(b_num_windows, 1, 1, 1, self.num_patches)

            if self.use_extra_keys:
                if extra_mask is None:
                    extra_mask = full_like(parts_mask, fill_value=mask_values['unmasked'])
                used_mask = cat((extra_mask, parts_mask), dim=-2)
            else:
                used_mask = parts_mask
            # used_mask.shape: [batch*Nwin, 1, 1, 1?2, winH*winW]

            if is_drop:
                dmask = empty_like(used_mask, dtype=torch_bool).bernoulli_(p=self.dropout)
                used_mask.masked_fill_(dmask, mask_values['masked'])

            score = score.flatten(-2)
            score = self.renorm(score, mask=used_mask.flatten(-2))
        # score.shape: [batch*Nwins, outwinH*outwinW, num_heads, winH*winW?*2]

        value = next(keys_i)
        value = value.view(b_num_windows, self.num_heads, -1, self.head_dim)
        # value.shape: [batch*Nwins, num_heads, winH*winW?*2, head_dim]

        output = einsum('...hid,...ohi->...ohd', value, score)
        # output.shape: [batch*Nwins, outwinH*outwinW, num_heads, head_dim]

        output = output.view(b_wic_win + self.projective_size + (-1,))
        # output.shape: [batch, win_in_cols, win_in_rows, outwinH, outwinW, num_heads*head_dim]

        output = self.finish(output)
        if self.activation:
            output = self.activation(output)

        # If resized
        #   output.shape: [batch, win_in_cols*outwinH, win_in_rows*outwinW, num_heads*head_dim]
        #   new_mask.shape: [batch, win_in_cols*outwinH, win_in_rows*outwinW]
        # If not resized
        #   output.shape: [batch, H, W, num_heads*head_dim]
        #   new_mask.shape: [batch, H, W]

        if not self.recalc_mask:
            if not is_drop:
                result = (join_windows(output, padding=padding)[0], mask)
            else:
                new_mask = used_mask.view(b_wic_win + (1, -1))
                new_mask = new_mask.amax(dim=-1, keepdim=True)
                new_mask = new_mask.expand(b_wic_win + self.projective_size)
                if parts_mask is not None:
                    new_mask = new_mask.minimum(parts_mask.view(b_wic_win + self.projective_size))
                result = join_windows(output, new_mask, padding=padding)
        else:
            if self.resize:
                padding = None

            # parts_mask.shape: [batch, win_in_cols, win_in_rows, winH, winW]
            # used_mask.shape: [batch*Nwin, 1, 1, 1?2, winH*winW]
            if used_mask is not None:
                new_mask = used_mask.view(b_wic_win + (1, -1))
                new_mask = new_mask.amax(dim=-1, keepdim=True)
                new_mask = new_mask.expand(b_wic_win + self.projective_size)
                result = join_windows(output, new_mask, padding=padding)
            else:
                result = join_windows(output, padding=padding)

        if self.return_keys:
            result += (keys, used_mask)

        return result


class HRIUN(Module):
    def __init__(self, window_size, channels, shifted=False, num_heads=1,
                 mid_activation=None, activation=None,
                 technique=None, mode=None,
                 scale=None, dropout=None,
                 hriu_factor=2, mlp_factor=2):
        super().__init__()

        window_size = tuple_n(2, window_size, 'window_size')
        head_dim = channels * hriu_factor // num_heads
        shift = tuple(x // 2 for x in window_size) if shifted else 0

        self.norm_hriu = InstanceNorm(norm_dims, affine_shape=channels)
        self.hriu = HRIU(
            window_size=window_size, channels=channels, shift=shift,
            projective_size=None, num_heads=num_heads,
            head_dim=head_dim, output_channels=None,
            mid_activation=mid_activation, activation=activation,
            technique=technique, mode=mode, scale=scale, dropout=dropout)
        self.norm_mlp = InstanceNorm(norm_dims, affine_shape=channels)
        self.mlp = MLP(in_channels=channels, factor=mlp_factor, activation=activation)

    def forward(self, input, mask=None, extra_keys=None, extra_mask=None):
        shortcut = x = input
        x = self.norm_hriu(x)
        res = self.hriu(x, mask, extra_keys, extra_mask)
        x, x_mask = res[:2]
        x += shortcut
        x_mask = union_mask(x_mask, mask)

        shortcut = x
        x = self.norm_mlp(x)
        x = self.mlp(x)
        x += shortcut

        return (x, x_mask) + res[2:]


class HRIUSca(Module):
    def __init__(self, window_size, channels, projective_size=None,
                 output_channels=None, num_heads=1,
                 mid_activation=None, activation=None,
                 mode=None,
                 scale=None, dropout=None,
                 hriu_factor=2, mlp_factor=2):
        super().__init__()

        window_size = tuple_n(2, window_size, 'window_size')
        head_dim = channels * hriu_factor // num_heads
        projective_size = projective_size or tuple(x // 2 for x in window_size)
        output_channels = output_channels or channels * 2

        self.norm_hriu = InstanceNorm(norm_dims, affine_shape=channels)

        self.hriu = HRIU(
            window_size=window_size, channels=channels, shift=0,
            projective_size=projective_size, num_heads=num_heads,
            head_dim=head_dim, output_channels=output_channels,
            mid_activation=mid_activation, activation=activation,
            technique=None, mode=mode, scale=scale, dropout=dropout)

        self.norm_mlp = InstanceNorm(norm_dims, affine_shape=output_channels)

        self.mlp = MLP(in_channels=output_channels, out_channels=output_channels,
                       factor=mlp_factor, activation=activation)

    def forward(self, input, mask=None, extra_keys=None, extra_mask=None):
        shortcut = x = input
        x = self.norm_hriu(x)
        res = self.hriu(x, mask, extra_keys, extra_mask)
        x, x_mask = res[:2]

        average, av_mask = masked_average2D(shortcut, mask, mean_channels=True, keepdim=True)
        x += average
        x_mask = union_mask(x_mask, av_mask)

        shortcut = x
        x = self.norm_mlp(x)
        x = self.mlp(x)
        x += shortcut

        return (x, x_mask) + res[2:]
