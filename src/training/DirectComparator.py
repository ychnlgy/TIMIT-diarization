import torch

class DirectComparator(torch.nn.Module):

    def __init__(self, layers):
        super().__init__()
        self.net = torch.nn.Sequential(*layers)

    def forward(self, X):
        v1 = self.net(X[:,0])
        v2 = self.net(X[:,1])
        w1 = self.net(X[:,2])

        sm = torch.nn.functional.cosine_similarity(v1, v2, dim=1)
        d1 = torch.nn.functional.cosine_similarity(v1, w1, dim=1)
        d2 = torch.nn.functional.cosine_similarity(v2, w1, dim=1)
        return sm, d1, d2

    def loss(self, X):
        sm, d1, d2 = self.forward(X)
        return (d1+d2-sm).mean()

    def loss_abs(self, X):
        sm, d1, d2 = self.forward(X)
        return ((1-sm)**2 + (-1-d1)**2 + (-1-d2)**2).sum()

    def score(self, X):
        sm, d1, d2 = self.forward(X)
        return (sm > d1).float().sum().item()
        
