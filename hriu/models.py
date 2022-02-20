# Simple models classes

from torch.nn import Module, ModuleList, Linear

from .utils import tuple_n, prod
from .windows import SpanningWindows
from .mask_tools import masked_average2D
from .base import HRIUN, HRIUSca


class HRIUclassifier(Module):
    def __init__(self, num_classes, pre, blocks, post,
                 window_size=4, num_heads=2, factor=2, projective_size=None,
                 patch=2, patch_stride=None,
                 activation=None,
                 dropout=None):
        super().__init__()

        in_channels = 3
        self.num_classes = num_classes

        window_size = tuple_n(2, window_size, 'window_size')
        self.activation = activation

        if projective_size is not None:
            projective_size = tuple_n(2, projective_size, 'projective_size')
            channel_multiplier = prod(window_size) // prod(projective_size) // 2
        else:
            projective_size = tuple(max(x // 2, 1) for x in window_size)
            channel_multiplier = 2

        patch = tuple_n(2, patch, 'patch')

        if patch_stride is not None:
            patch_stride = tuple_n(2, patch_stride, 'patch_stride')
        else:
            patch_stride = patch

        self.split_patches = SpanningWindows(patch, patch_stride)

        channels = prod(patch) * in_channels
        self.pre = ModuleList()
        for n in pre:
            self.pre.append(Linear(channels, n))
            channels = n

        self.blocks = ModuleList()
        for n_layers in blocks:
            for i in range(n_layers):
                self.blocks.append(
                    HRIUN(window_size=window_size, channels=channels,
                          shifted=False, num_heads=num_heads,
                          activation=activation, dropout=dropout,
                          technique=None,
                          hriu_factor=factor, mlp_factor=factor)
                )
                self.blocks.append(
                    HRIUN(window_size=window_size, channels=channels,
                          shifted=True, num_heads=num_heads,
                          activation=activation, dropout=dropout,
                          technique=None,
                          hriu_factor=factor, mlp_factor=factor)
                )
            self.blocks.append(
                HRIUSca(window_size=window_size, channels=channels,
                        projective_size=projective_size,
                        output_channels=channels * channel_multiplier,
                        num_heads=num_heads,
                        activation=activation, dropout=dropout,
                        hriu_factor=factor, mlp_factor=factor)
            )
            channels *= channel_multiplier

        self.post = ModuleList()
        for n in post:
            self.post.append(Linear(channels, n))
            channels = n
        self.final = Linear(channels, self.num_classes)

    def forward(self, input, mask=None):
        x, mask, padding = self.split_patches(input, mask)
        x = x.flatten(-3)
        if mask is not None:
            mask = mask.amin(dim=(-1, -2))

        for layer in self.pre:
            x = layer(x)
            if self.activation:
                x = self.activation(x)

        for layer in self.blocks:
            x, mask = layer(x, mask)

        x, _ = masked_average2D(x, mask)

        for layer in self.post:
            x = layer(x)
            if self.activation:
                x = self.activation(x)

        return self.final(x)
