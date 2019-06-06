import torch

class GlobalAveragePool(torch.nn.Module):

    def forward(self, X):
        return X.view(X.size(0), X.size(1), -1).mean(dim=-1)
