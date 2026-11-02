import torch
import torch.distributions as dist


class DiagStudentTPrior(dist.Distribution):
    arg_constraints = {}
    support = dist.constraints.real
    has_rsample = False

    def __init__(self, df: float, loc: torch.Tensor, scale: torch.Tensor, validate_args=None):
        self.df = torch.as_tensor(df, dtype=loc.dtype, device=loc.device)
        base = dist.StudentT(df=self.df, loc=loc, scale=scale)
        self._dist = dist.Independent(base, 1)  # Treat as event_shape=[dim]
        super().__init__(validate_args=validate_args)

    @property
    def event_shape(self):
        return self._dist.base_dist.loc.shape

    def sample(self, sample_shape=torch.Size()):
        return self._dist.sample(sample_shape)

    def log_prob(self, value: torch.Tensor):
        return self._dist.log_prob(value)


class DiagGaussianPrior(dist.Distribution):
    arg_constraints = {}
    support = dist.constraints.real
    has_rsample = True

    def __init__(self, loc: torch.Tensor, scale: torch.Tensor, validate_args=None):
        base = dist.Normal(loc=loc, scale=scale)
        self._dist = dist.Independent(base, 1)
        super().__init__(validate_args=validate_args)

    @property
    def event_shape(self):
        return self._dist.base_dist.loc.shape

    def sample(self, sample_shape=torch.Size()):
        return self._dist.sample(sample_shape)

    def log_prob(self, value: torch.Tensor):
        return self._dist.log_prob(value)


def make_default_prior(acts: torch.Tensor, prior_type: str = "student_t", df: float = 5.0,
                       whitening: bool = True):
    """
    Creates a prior distribution fitted to the activation samples.

    If `whitening=True` the activations are assumed to already be whitened
    (mean ~ 0, std ~ 1) and standard parameters are used.
    """
    dim = acts.shape[1]
    device = acts.device

    if whitening:
        loc = torch.zeros(dim, device=device)
        scale = torch.ones(dim, device=device)
    else:
        loc = acts.mean(dim=0)
        scale = acts.std(dim=0).clamp_min(1e-6)

    if prior_type == "student_t":
        return DiagStudentTPrior(df=df, loc=loc, scale=scale)
    elif prior_type == "gaussian":
        return DiagGaussianPrior(loc=loc, scale=scale)
    else:
        raise ValueError(f"Unknown prior_type: {prior_type}. Must be 'student_t' or 'gaussian'")
