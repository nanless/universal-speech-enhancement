import abc

import numpy as np
import torch

from src.models.components.sgmse.util.registry import Registry

PredictorRegistry = Registry("Predictor")


class Predictor(abc.ABC):
    """The abstract class for a predictor algorithm."""

    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__()
        self.sde = sde
        self.rsde = sde.reverse(score_fn)
        self.score_fn = score_fn
        self.probability_flow = probability_flow

    @abc.abstractmethod
    def update_fn(self, x, t, *args):
        """One update of the predictor.

        Args:
            x: A PyTorch tensor representing the current state
            t: A Pytorch tensor representing the current time step.
            *args: Possibly additional arguments, in particular `y` for OU processes

        Returns:
            x: A PyTorch tensor of the next state.
            x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
        """
        pass

    def debug_update_fn(self, x, t, *args):
        raise NotImplementedError(f"Debug update function not implemented for predictor {self}.")


@PredictorRegistry.register("euler_maruyama")
class EulerMaruyamaPredictor(Predictor):
    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__(sde, score_fn, probability_flow=probability_flow)

    def update_fn(self, x, t, *args, **kwargs):
        dt = -1.0 / self.rsde.N
        z = torch.randn_like(x)
        f, g = self.rsde.sde(x, t, *args, **kwargs)
        x_mean = x + f * dt
        if g.ndim < x.ndim:
            g = g.view(*g.size(), *((1,) * (x.ndim - g.ndim)))
        x = x_mean + g * np.sqrt(-dt) * z
        return x, x_mean


@PredictorRegistry.register("reverse_diffusion")
class ReverseDiffusionPredictor(Predictor):
    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__(sde, score_fn, probability_flow=probability_flow)

    def update_fn(self, x, t, *args, **kwargs):
        f, g = self.rsde.discretize(x, t, *args, **kwargs)
        z = torch.randn_like(x)
        x_mean = x - f
        if g.ndim < x.ndim:
            g = g.view(*g.size(), *((1,) * (x.ndim - g.ndim)))
        x = x_mean + g * z
        return x, x_mean


@PredictorRegistry.register("none")
class NonePredictor(Predictor):
    """An empty predictor that does nothing."""

    def __init__(self, *args, **kwargs):
        pass

    def update_fn(self, x, t, *args, **kwargs):
        return x, x
