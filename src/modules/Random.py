import torch

class Random(torch.nn.Module):
    '''

    Sets a proportion p of the input into random numbers
    sampled uniformly from [a, b].

    Returns the original input if not training.

    '''

    def __init__(self, p, a, b):
        super().__init__()
        self.p = p
        self.a = a
        self.b = b

    def forward(self, X):
        if self.training:
            r1 = torch.rand_like(X)
            r2 = torch.rand_like(X)*(self.b-self.a)+self.a
            I = (r1 < self.p).float()
            X = X*(1-I) + I*r2
        return X
