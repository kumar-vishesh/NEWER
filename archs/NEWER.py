# Necassary imports
import torch
import torch.nn as nn
from .SIREN import SIREN


class NEWER(nn.Module):
    """
    2 independent SIRENs for φ and α.
    Each predicts n_wavelets outputs given (c, t) coordinates.
    """

    def __init__(self, in_dim=2, hidden_dim=16, hidden_layers=4, n_wavelets=1000, omega_0=30, sigma=0.005):
        super().__init__()
        phi_offsets = torch.linspace(-1.0, 1.0, n_wavelets).view(1, n_wavelets)
        self.register_buffer("phi_offsets", phi_offsets)
        self.register_buffer("sigma", torch.tensor(sigma))

        self.siren_phi = SIREN(in_dim, hidden_dim, hidden_layers, n_wavelets, omega_0)
        self.omega = nn.Parameter(250 + 10 * torch.randn(n_wavelets))
        # self.siren_amplitude = SIREN(in_dim - 1, hidden_dim, hidden_layers - 1, n_wavelets, omega_0)
        self.amplitude = nn.Parameter(torch.randn(n_wavelets) / n_wavelets)
        self.siren_alpha = SIREN(in_dim, hidden_dim, hidden_layers - 1, n_wavelets, omega_0)

    def forward(self, x):
        t = x[:, 1:2]
        c = x[:, 0:1]
        phi = torch.sin(self.siren_phi(x))
        omega = torch.nn.functional.softplus(self.omega)
        alpha = torch.pi * torch.sin(self.siren_alpha(x))
        # amp = self.siren_amplitude(c)
        amp = self.amplitude
        gauss = torch.exp(-((t - (phi + self.phi_offsets)) ** 2) / (2 * self.sigma**2))
        wave = amp * gauss * torch.cos(omega * (t - phi) - alpha)
        wave = wave.sum(dim=-1, keepdim=True)
        return wave
