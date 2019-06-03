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
        return self.ac(self.bk(X)*self.w8 + self.sc(X)*self.w8)
