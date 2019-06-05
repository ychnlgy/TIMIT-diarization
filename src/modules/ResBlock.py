import torch

IDENTITY = torch.nn.Sequential()

class ResBlock(torch.nn.Module):

    '''

    Returns the residual transformation of X.

    The parameter "weight" balances the outputs. Typically, we use weight=1
    to sum the shortcut with the branch transformation. Other times we wish
    to maintain the magnitude of inputs, so we average them with weight=0.5.

    '''

    def __init__(self, block, weight=1, shortcut=IDENTITY, activation=IDENTITY):
        super(ResBlock, self).__init__()
        self.bk = block
        self.sc = shortcut
        self.ac = activation
        self.w8 = weight
        
    def forward(self, X):
        h0 = self.sc(X)
        h1 = self.bk(X)
        h2 = (h0 + h1) * self.w8
        return self.ac(h2)
