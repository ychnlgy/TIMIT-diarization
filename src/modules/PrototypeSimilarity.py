import torch, math

from .CosineSimilarity import CosineSimilarity

DEFAULT = CosineSimilarity(dim=2)

class PrototypeSimilarity(torch.nn.Module):

    def __init__(self, features, classes, similarity=DEFAULT):
        super().__init__()
        self.C = classes
        self.D = features
        self.weight = torch.nn.Parameter(torch.zeros(1, classes, features))
        self.similarity = similarity

        self.reset_parameters()

    def forward(self, X):
        '''

        Input:
            X - torch Tensor of shape (N, D, *), input features.

        Output:
            X' - torch Tensor of shape (N, C, *), the cosine similarity
                between the feature vector and each C prototypes.

        '''
        X = X.unsqueeze(1)
        e = len(X.shape) - len(self.weight.shape)
        P = self.weight.view(*self.weight.shape, *([1]*e))
        return self.similarity(X, P)

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
