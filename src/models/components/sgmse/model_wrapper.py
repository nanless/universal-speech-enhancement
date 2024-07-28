from math import ceil

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.components.sgmse import sampling
from src.models.components.sgmse.backbones import BackboneRegistry
from src.models.components.sgmse.sdes import SDERegistry
from src.models.components.sgmse.util.other import pad_spec


def get_window(window_type, window_length):
    if window_type == "sqrthann":
        return torch.sqrt(torch.hann_window(window_length, periodic=True))
    elif window_type == "hann":
        return torch.hann_window(window_length, periodic=True)
    else:
        raise NotImplementedError(f"Window type {window_type} not implemented!")


class ScoreModel(nn.Module):
    def __init__(
        self,
        backbone: str = "ncsnpp",
        sde: str = "ouve",
        t_eps: float = 3e-2,
        mode="regen-joint-training",
        condition="both",
        loss_type: str = "mse",
        n_fft=510,
        hop_length=128,
        num_frames=256,
        window="hann",
        spec_factor=0.15,
        spec_abs_exponent=0.5,
        sde_input="denoised",
        predictor="reverse_diffusion",
        corrector="none",
    ):
        super().__init__()
        if condition == "both":
            input_channels = 6
        else:
            input_channels = 4
        self.score_net = (
            BackboneRegistry.get_by_name(backbone)(input_channels=input_channels)
            if backbone != "none"
            else None
        )

        # Initialize SDE
        sde_cls = SDERegistry.get_by_name(sde)
        self.sde = sde_cls()
        self.t_eps = t_eps

        self.condition = condition
        self.mode = mode

        self.loss_type = loss_type
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.num_frames = num_frames

        self.window = get_window(window, self.n_fft)
        self.windows = {}
        self.stft_kwargs = {
            "n_fft": n_fft,
            "hop_length": hop_length,
            "window": self.window,
            "center": True,
            "return_complex": True,
        }
        self.istft_kwargs = {
            "n_fft": n_fft,
            "hop_length": hop_length,
            "window": self.window,
            "center": True,
        }
        self.spec_factor = spec_factor
        self.spec_abs_exponent = spec_abs_exponent

        self.target_len = (self.num_frames - 1) * self.hop_length

        self.sde_input = sde_input

        self.predictor = predictor

        self.corrector = corrector

    def spec_fwd(self, spec):
        if self.spec_abs_exponent != 1:
            e = self.spec_abs_exponent
            spec = spec.abs() ** e * torch.exp(1j * spec.angle())
        return spec * self.spec_factor

    def spec_back(self, spec):
        spec = spec / self.spec_factor
        if self.spec_abs_exponent != 1:
            e = self.spec_abs_exponent
            spec = spec.abs() ** (1 / e) * torch.exp(1j * spec.angle())
        return spec

    def _get_window(self, x):
        """Retrieve an appropriate window for the given tensor x, matching the device.

        Caches the retrieved windows so that only one window tensor will be allocated per device.
        """
        window = self.windows.get(x.device, None)
        if window is None:
            window = self.window.to(x.device)
            self.windows[x.device] = window
        return window

    def stft(self, sig):
        window = self._get_window(sig)
        return torch.stft(sig, **{**self.stft_kwargs, "window": window})

    def istft(self, spec, length=None):
        window = self._get_window(spec)
        return torch.istft(spec, **{**self.istft_kwargs, "window": window, "length": length})

    def _loss(self, err):
        if self.loss_type == "mse":
            losses = torch.square(err.abs())
            loss = torch.mean(0.5 * torch.sum(losses.reshape(losses.shape[0], -1), dim=-1))

        elif self.loss_type == "mae":
            losses = err.abs()
            loss = torch.mean(0.5 * torch.sum(losses.reshape(losses.shape[0], -1), dim=-1))

        return loss

    def forward_score(self, x, t, score_conditioning, sde_input):
        dnn_input = torch.cat([x] + score_conditioning, dim=1)  # b,n_input*d,f,t
        score = -self.score_net(dnn_input, t)
        std = self.sde._std(t, y=sde_input)
        if std.ndim < sde_input.ndim:
            std = std.view(*std.size(), *((1,) * (sde_input.ndim - std.ndim)))
        return score

    def forward(self, x, t, score_conditioning, sde_input):
        score = self.forward_score(x, t, score_conditioning, sde_input)
        return score

    def train_step(self, batch):
        x, y = batch["clean"], batch["perturbed"]
        if "fake" in batch:
            y_denoised = batch["fake"]

        current_len = x.size(-1)
        pad = max(self.target_len - current_len, 0)
        if pad == 0:
            # extract random part of the audio file
            start = int(np.random.uniform(0, current_len - self.target_len))
            x = x[..., start : start + self.target_len]
            y = y[..., start : start + self.target_len]
            if "fake" in batch:
                y_denoised = y_denoised[..., start : start + self.target_len]
        else:
            # pad audio if the length T is smaller than num_frames
            x = F.pad(x, (pad // 2, pad // 2 + (pad % 2)), mode="constant")
            y = F.pad(y, (pad // 2, pad // 2 + (pad % 2)), mode="constant")
            if "fake" in batch:
                y_denoised = F.pad(y_denoised, (pad // 2, pad // 2 + (pad % 2)), mode="constant")

        x = self.spec_fwd(self.stft(x)).unsqueeze(1)
        y = self.spec_fwd(self.stft(y)).unsqueeze(1)
        if "fake" in batch:
            y_denoised = self.spec_fwd(self.stft(y_denoised)).unsqueeze(1)

        sde_target = x
        if self.sde_input == "denoised" and "fake" in batch:
            sde_input = y_denoised
        elif self.sde_input == "noisy":
            sde_input = y
        else:
            raise NotImplementedError(
                f"Don't know the sde input you have wished for: {self.sde_input}"
            )
        # Forward process
        t = torch.rand(x.shape[0], device=x.device) * (self.sde.T - self.t_eps) + self.t_eps
        mean, std = self.sde.marginal_prob(sde_target, t, sde_input)
        z = torch.randn_like(x)  # i.i.d. normal distributed with var=0.5
        if std.ndim < y.ndim:
            std = std.view(*std.size(), *((1,) * (y.ndim - std.ndim)))
        sigmas = std
        perturbed_data = mean + sigmas * z

        # Score estimation
        if self.condition == "noisy":
            score_conditioning = [y]
        elif self.condition == "denoised" and "fake" in batch:
            score_conditioning = [y_denoised]
        elif self.condition == "both" and "fake" in batch:
            score_conditioning = [y, y_denoised]
        else:
            raise NotImplementedError(
                f"Don't know the conditioning you have wished for: {self.condition}"
            )

        score = self.forward_score(perturbed_data, t, score_conditioning, sde_input)
        err = score * sigmas + z

        loss = self._loss(err)

        return loss

    def get_pc_sampler(self, predictor_name, corrector_name, y, N=None, minibatch=None, **kwargs):
        N = self.sde.N if N is None else N
        sde = self.sde.copy()
        sde.N = N

        kwargs = {"eps": self.t_eps, **kwargs}
        if minibatch is None:
            return sampling.get_pc_sampler(
                predictor_name, corrector_name, sde=sde, score_fn=self, y=y, **kwargs
            )
        else:
            M = y.shape[0]

            def batched_sampling_fn():
                samples, ns = [], []
                for i in range(int(ceil(M / minibatch))):
                    y_mini = y[i * minibatch : (i + 1) * minibatch]
                    sampler = sampling.get_pc_sampler(
                        predictor_name, corrector_name, sde=sde, score_fn=self, y=y_mini, **kwargs
                    )
                    sample, n = sampler()
                    samples.append(sample)
                    ns.append(n)
                samples = torch.cat(samples, dim=0)
                return samples, ns

            return batched_sampling_fn

    def get_ode_sampler(self, y, N=None, minibatch=1, **kwargs):
        N = self.sde.N if N is None else N
        sde = self.sde.copy()
        sde.N = N

        kwargs = {"eps": self.t_eps, **kwargs}
        if minibatch is None:
            return sampling.get_ode_sampler(sde, self, y=y, **kwargs)
        else:
            M = y.shape[0]

            def batched_sampling_fn():
                samples, ns = [], []
                for i in range(int(ceil(M / minibatch))):
                    y_mini = y[i * minibatch : (i + 1) * minibatch]
                    sampler = sampling.get_ode_sampler(sde, self, y=y_mini, **kwargs)
                    sample, n = sampler()
                    samples.append(sample)
                    ns.append(n)
                samples = torch.cat(samples, dim=0)
                return samples, ns

            return batched_sampling_fn

    def sample(
        self,
        batch,
        sampler_type="pc",
        N=50,
        corrector_steps=1,
        snr=0.5,
    ):
        y = batch["perturbed"]
        if "fake" in batch:
            y_denoised = batch["fake"]
        T_orig = y.size(1)

        Y = self.spec_fwd(self.stft(y)).unsqueeze(1)
        if "fake" in batch:
            Y_denoised = self.spec_fwd(self.stft(y_denoised)).unsqueeze(1)
        Y = pad_spec(Y)
        if "fake" in batch:
            Y_denoised = pad_spec(Y_denoised)

        # Conditioning
        if self.condition == "noisy":
            score_conditioning = [Y]
        elif self.condition == "denoised" and "fake" in batch:
            score_conditioning = [Y_denoised]
        elif self.condition == "both" and "fake" in batch:
            score_conditioning = [Y, Y_denoised]
        else:
            raise NotImplementedError(
                f"Don't know the conditioning you have wished for: {self.condition}"
            )

        if self.sde_input == "denoised" and "fake" in batch:
            sde_input = Y_denoised
        elif self.sde_input == "noisy":
            sde_input = Y
        else:
            raise NotImplementedError(
                f"Don't know the sde input you have wished for: {self.sde_input}"
            )

        # Reverse process
        if sampler_type == "pc":
            sampler = self.get_pc_sampler(
                self.predictor,
                self.corrector,
                sde_input,
                N=N,
                corrector_steps=corrector_steps,
                snr=snr,
                intermediate=False,
                conditioning=score_conditioning,
            )
        elif sampler_type == "ode":
            sampler = self.get_ode_sampler(sde_input, N=N, conditioning=score_conditioning)
        else:
            print(f"{sampler_type} is not a valid sampler type!")
        sample, nfe = sampler()
        y_score_enhanced = self.istft(self.spec_back(sample.squeeze(1)), T_orig)
        if self.sde_input == "denoised" and "fake" in batch:
            batch["fake_sde_enhanced"] = y_score_enhanced
        elif self.sde_input == "noisy":
            batch["enhanced"] = y_score_enhanced
        else:
            raise NotImplementedError(
                f"Don't know the sde input you have wished for: {self.sde_input}"
            )
        return batch
