import torch

class Comparator(torch.nn.Sequential):

    def forward(self, X):
        v1 = super().forward(X[:,0])
        v2 = super().forward(X[:,1])
        return torch.nn.functional.cosine_similarity(v1, v2, dim=1)

    def score(self, Yh, Y):
        assert Yh.size() == Y.size()
        pred = Yh > 0
        true = Y > 0
        return (pred == true).float().sum().item()
