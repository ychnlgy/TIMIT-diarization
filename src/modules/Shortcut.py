import torch

class Shortcut(torch.nn.Module):

    '''

    The original shortcut connection for stride 2 was simply:

    X#X#X#X#X#X#X#X#X#X#
    ####################
    X#X#X#X#X#X#X#X#X#X#
    ####################
    X#X#X#X#X#X#X#X#X#X#
    ####################
    X#X#X#X#X#X#X#X#X#X#
    ####################

    This shortcut variation is copied from the PyTorch Shake-Shake
    implementation, which computes the concatenation of the original with:

    ####################
    #X#X#X#X#X#X#X#X#X#X
    ####################
    #X#X#X#X#X#X#X#X#X#X
    ####################
    #X#X#X#X#X#X#X#X#X#X
    ####################
    #X#X#X#X#X#X#X#X#X#X

    Note that in our implementation, average pooling is not averaging anything
    since the kernel size is 1. Instead, it selects the single element per stride.

    '''

    def __init__(self, in_c, out_c, stride, act):
        super().__init__()
        assert out_c & 2 == 0
        self.ak = act
        self.c1 = torch.nn.Sequential(
            torch.nn.AvgPool2d(1, stride=stride),
            torch.nn.Conv2d(in_c, out_c//2, 1, stride=1, bias=False)
        )
        self.c2 = torch.nn.Sequential(
            torch.nn.ZeroPad2d((-1, 1, -1, 1)),
            torch.nn.AvgPool2d(1, stride=stride),
            torch.nn.Conv2d(in_c, out_c//2, 1, stride=1, bias=False)
        )
        self.bn = torch.nn.BatchNorm2d(out_c)

    def forward(self, X):
        h0 = self.ak(X)
        h1 = self.c1(h0)
        h2 = self.c2(h0)
        hx = torch.cat([h1, h2], dim=1)
        return self.bn(hx)
