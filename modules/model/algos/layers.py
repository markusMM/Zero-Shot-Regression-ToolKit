import torch
from torch import nn
from torch.functional import F
from torch.distributions import Bernoulli, Exponential, Normal
from warnings import warn


class ExpMoG(nn.Module):

    def __init__(
        self,
        n_units: int,
        n_clusters: int = 12
    ):
        self.sigmas = torch.rand(n_units, n_clusters)
        self.mues = nn.Parameter(torch.rand(n_units, n_clusters))
        self.lamb = nn.Parameter(torch.rand(n_units))
        self.llh = -torch.inf

    def forward(self, x):
        p = torch.expm1(Exponential(torch.sigmoid(self.lamb)).log_prob(self.mues))
        dist = Normal(
            self.mues,
            p * torch.sigma(self.sigmas)
        )
        llh = torch.log1p(torch.expm1(dist.log_prob(x[:, :, None])).sum(-1))
        post = llh * p / (llh * p).sum()
        self.llh = llh.sum()
        return x * post


class ADBUDGResponse(nn.Module):
    r"""
    Perform ADBUDG acivation.

    This function performs a parameterized ADBUDG transformation:

    .. math:: f(x) = \frac{x^\gamma}{(\rho + x^\gamma)}

    with `gamma` and `rho` as constraint parameters:

    .. math:: \rho(p_1) = ReLU(p_1) + \rho_min

    with `rho_min` being strictly greater than 0.

    .. math:: \gamma(p_2) = \gamma_min + \left(\gamma_max-\gamma_max\right)Sigmoid(p_2)

    with `gamma_min` being strictly greater or equal to 1.

    If `magnitude_response` was set, the input's absolute value is taken and the result
    is multiplied by it\'s original sign:

    .. math:: f(x) = sign(x) \frac{|x|^\gamma}{(\rho + |x|^\gamma)}

    :param n_units: Number of variables in the corresponding layer.
    :param rho_min: Minimum value for constant.
    :param gam_min: Minimum value for the exponent.
    :param gam_max: Maximum value for the exponent.
    :param magnitude_response: Rather to preserve the sign and magnitude of the input.
    """

    def __init__(
            self,
            n_units: int,
            rho_min: float = 1.0,
            gam_min: float = 1.1,
            gam_max: float = 3.0,
            magnitude_response: bool = False
    ):
        super().__init__()

        if gam_min < 1:
            warn(f'Unstable minimum for gamma: {gam_min}')
            warn('Setting to 1!')
            gam_min = 1.0

        if gam_min <= 0:
            warn(f'Unstable minimum for gamma: {gam_min}')
            warn('Setting to 1!')
            gam_min = 1.0

        self.rho = nn.Parameter(1 + torch.rand(n_units))
        self.gam = nn.Parameter(1 + torch.rand(n_units))
        self.rho_min = rho_min
        self.gam_min = gam_min
        self.gam_max = gam_max
        self.mag_res = magnitude_response

    def forward(self, x):
        rho = F.relu(self.rho) + self.rho_min
        gam = self.gam_min + (self.gam_max - self.gam_min) * F.sigmoid(self.gam)

        def tf(v):
            return v**gam / (rho + v**gam)

        if not self.mag_res:
            x = torch.relu(x)
            x = tf(x)
        else:
            sign = x.sign()
            x = x.abs()
            x = sign * tf(x)
        return x


class BerDrop(nn.Module):

    def __init__(
            self,
            pies: float = .05
    ):
        super().__init__()
        self.pies = nn.Parameter(
            torch.tensor(pies)
        )

    def forward(self, x):
        z = Bernoulli(
            torch.sigmoid(self.pies)
        ).sample(x.shape)
        return x * z
