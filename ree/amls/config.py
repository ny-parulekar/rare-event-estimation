from dataclasses import dataclass


@dataclass
class AMLSConfig:
    """
    Configuration for GA-AMLS (gradient-assisted adaptive multilevel splitting).
    Defaults are the final tuned values used for all experiments in the paper.
    """

    # AMLS
    n_particles: int = 2000
    n_levels: int = 500          # max number of levels (loop safety cap)
    rho: float = 0.6             # fraction of particles kept at each level
    mh_steps: int = 900          # MCMC steps per level
    burn_percentage: float = 0.14  # fraction of mh_steps used for step-size adaptation
    n_burn: int = None           # explicit burn-in steps (overrides burn_percentage)

    # MALA sampler
    h: float = 1e-3              # Langevin step size
    target_acc: float = 0.57     # target acceptance rate for adaptation

    # Prior / score function
    prior_type: str = "student_t"  # "student_t" or "gaussian"
    beta: float = 50.0             # LogSumExp smoothness (only used by smooth score mode)

    # Misc
    seed: int = 0
    device: str = "cuda"
