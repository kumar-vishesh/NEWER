# Necassary imports
import torch
import torch.nn as nn
import numpy as np


class ComplexGaborLayer(nn.Module):
    """
    Implicit representation with complex Gabor nonlinearity

    Inputs;
        in_dim: Input dim
        out_dim; Output dim
        bias: if True, enable bias for the linear operation
        is_first: Legacy SIREN parameter
        omega_0: Legacy SIREN parameter
        omega0: Frequency of Gabor sinusoid term
        sigma0: Scaling of Gabor Gaussian term
        trainable: If True, omega and sigma are trainable parameters
    """

    def __init__(self, in_dim, out_dim, bias=True, is_first=False, omega0=10.0, sigma0=40.0, trainable=False):
        super().__init__()
        self.omega_0 = omega0
        self.scale_0 = sigma0
        self.is_first = is_first

        self.in_dim = in_dim

        if self.is_first:
            dtype = torch.float
        else:
            dtype = torch.cfloat

        # Set trainable parameters if they are to be simultaneously optimized
        self.omega_0 = nn.Parameter(self.omega_0 * torch.ones(1), trainable)
        self.scale_0 = nn.Parameter(self.scale_0 * torch.ones(1), trainable)

        self.linear = nn.Linear(in_dim, out_dim, bias=bias, dtype=dtype)

    def forward(self, input):
        lin = self.linear(input)
        omega = self.omega_0 * lin
        scale = self.scale_0 * lin

        return torch.exp(1j * omega - scale.abs().square())


class WIRE(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dim,
        hidden_layers,
        out_dim,
        first_omega_0=1.0,
        hidden_omega_0=10.0,
        scale=10.0,
    ):
        super().__init__()

        # All results in the paper were with the default complex 'gabor' nonlinearity
        self.nonlin = ComplexGaborLayer

        # Since complex numbers are two real numbers, reduce the number of
        # hidden parameters by 2
        hidden_dim = int(hidden_dim / np.sqrt(2))
        dtype = torch.cfloat
        self.complex = True
        self.wavelet = "gabor"

        # Legacy parameter
        self.pos_encode = False

        self.net = []
        self.net.append(
            self.nonlin(in_dim, hidden_dim, omega0=first_omega_0, sigma0=scale, is_first=True, trainable=False)
        )

        for i in range(hidden_layers - 1):
            self.net.append(self.nonlin(hidden_dim, hidden_dim, omega0=hidden_omega_0, sigma0=scale))

        final_linear = nn.Linear(hidden_dim, out_dim, dtype=dtype)
        with torch.no_grad():
            final_linear.weight.uniform_(-1e-4, 1e-4)
        self.net.append(final_linear)

        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        output = self.net(coords)

        if self.wavelet == "gabor":
            return output.real

        return output
