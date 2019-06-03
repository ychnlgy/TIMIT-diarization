import torch, math

from . import chebyshev, LagrangeBasis

class Activation(torch.nn.Module):

    '''

    The basic Chebyshev-Lagrange (CL) implementation.

    Since backpropagation of signals is not directly proportional
    to the output, we choose to initialize all parameters to 0.

    Recall the parameters represent the y-positions of the Chebyshev nodes.

    '''

    def __init__(self, input_size, n_degree, zeros=True):
        super().__init__()
        self.d = input_size
        self.n = n_degree + 1
        self.radius = self._calc_radius(self.n)
        self.basis = LagrangeBasis.create(
            chebyshev.get_nodes(self.n, -self.radius, self.radius)
        )
        self.weight = torch.nn.Parameter(
            torch.zeros(1, self.d, self.n, 1)
        )

        if not zeros:
            self.randomize_parameters()

    def forward(self, X):
        '''

        Input:
            X - torch Tensor of shape (N, D, *), input features.

        Output:
            X' - torch Tensor of shape (N, D, *), outputs.

        '''
        N = X.size(0)
        D = X.size(1)
        
        B = self.basis(X.view(N, D, -1)) # (N, D, n, -1)
        L = (self.weight * B).sum(dim=2)
        return L.view(X.size())

    # === PRIVATE ===

    def randomize_parameters(self):
        scale = math.sqrt(2.0/(self.d+self.n))
        self.weight.data.uniform_(-scale, scale)

    def _calc_radius(self, n):
        "Returns a radius for Chebyshev nodes such that x[0] = 1 and x[n+1] = -1."
        return 1.0/math.cos(math.pi/(2*n))
