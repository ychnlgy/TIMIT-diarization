import torch, math

class ChebyshevActivation(torch.nn.Module):

    '''

    Computes weighted Chebyshev polynomial (WCP).

    Recall it is done recursively:

        - T[0](x) = 1
        - T[1](x) = x
        - T[i](x) = 2xT[i-1](x) - T[i-2](x) {for i > 1}

    This operation is done in the operate() method.

    Recall the final output is:

        y = w[0]*T[0](x) + w[1]*T[1](x) + ... + w[n]*T[n](x)

    per feature.

    '''

    def __init__(self, input_size, n_degree):
        super().__init__()
        self.d = input_size
        self.n = n_degree + 1
        self.weight = torch.nn.Parameter(
            torch.zeros(1, self.d, self.n)
        )
        self._axes = None

    def forward(self, X):
        '''

        Input:
            X - torch Tensor of size (N, D, *), input features.

        Output:
            P - torch Tensor of size (N, D, *), polynomial output.

        '''
        B = self.operate(X, [], self.n)
        e = len(B.shape) - len(self.weight.shape)
        w = self.weight.view(1, self.d, *([1]*e), self.n)
        L = (w * B).sum(dim=-1)
        assert L.size() == X.size()
        return L

    def operate(self, X, H, stop):
        if len(H) < stop:
            
            if len(H) == 0:
                H.extend([torch.ones_like(X), X])
            else:
                H.append(H[-1]*2*X-H[-2])
            return self.operate(X, H, stop)
        else:
            return torch.stack(H, dim=-1)

    def reset_parameters(self):
        self.weight.data.zero_()
