import torch

from .ResBlock import ResBlock

class ResNet(torch.nn.Module):

    def __init__(self, block_constructor, shortcut_constructor, channels, block_depth):
        super().__init__()
        self.net = torch.nn.Sequential(*[
            self._construct_block(
                block_depth,
                block_constructor,
                shortcut_constructor,
                channels[i], channels[i+1]
            ) for i in range(len(channels)-1)
        ])

    def forward(self, X):
        return self.net(X)

    # === PRIVATE ===

    def _construct_block(self, block_depth, block_constructor, shortcut_constructor, in_c, out_c):
        out = []
        for i in range(block_depth):
            out.append(self._construct_one_block(block_constructor, shortcut_constructor, in_c, out_c))
            in_c = out_c
        return torch.nn.Sequential(*out)

    def _construct_one_block(self, block_constructor, shortcut_constructor, in_c, out_c):
        block = block_constructor(in_c, out_c)
        if out_c == in_c:
            return ResBlock(block)
        else:
            return ResBlock(block, shortcut=shortcut_constructor(in_c, out_c))
