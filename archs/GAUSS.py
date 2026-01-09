# Necassary imports
import torch
import torch.nn as nn


class GaussLayer(nn.Module):
    """
    Drop in replacement for SineLayer but with Gaussian non linearity
    """

    def __init__(self, in_dim, out_dim, bias=True, is_first=False, scale=10.0):
        """
        is_first, and omega_0 are not used.
        """
        super().__init__()
        self.in_dim = in_dim
        self.scale = scale
        self.is_first = is_first
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)

    def forward(self, input):
        return torch.exp(-((self.scale * self.linear(input)) ** 2))


class GAUSS(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dim,
        hidden_layers,
        out_dim,
        outermost_linear=True,
        scale=10.0,
    ):
        super().__init__()

        self.complex = False
        self.nonlin = GaussLayer

        self.net = []
        self.net.append(self.nonlin(in_dim, hidden_dim, is_first=True, scale=scale))

        for i in range(hidden_layers - 1):
            self.net.append(self.nonlin(hidden_dim, hidden_dim, is_first=False, scale=scale))

        if outermost_linear:
            if self.complex:
                dtype = torch.cfloat
            else:
                dtype = torch.float
            final_linear = nn.Linear(hidden_dim, out_dim, dtype=dtype)

            self.net.append(final_linear)
        else:
            self.net.append(self.nonlin(hidden_dim, out_dim, is_first=False, scale=scale))
        with torch.no_grad():
            final_linear.weight.uniform_(-1e-10, 1e-10)
        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        output = self.net(coords)

        return output
