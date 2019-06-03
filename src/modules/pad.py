import torch

class VectorPad(torch.nn.Module):

    def __init__(self, D):
        super().__init__()
        self.D = D

    def forward(self, X):
        N, D = X.size()
        Z = torch.zeros(N, self.D, device=X.device)
        Z[:,:D] = X
        return Z
