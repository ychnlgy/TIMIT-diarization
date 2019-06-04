import torch

class Repeat(torch.nn.Sequential):

    def __init__(self, constructor, repeat):
        super().__init__(
            *[constructor() for i in range(repeat)]
        )
