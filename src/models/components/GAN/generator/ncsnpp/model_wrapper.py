import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.components.sgmse.backbones.ncsnpp import NCSNpp
from src.models.components.sgmse.util.other import pad_spec


def get_window(window_type, window_length):
    if window_type == "sqrthann":
        return torch.sqrt(torch.hann_window(window_length, periodic=True))
    elif window_type == "hann":
        return torch.hann_window(window_length, periodic=True)
    else:
        raise NotImplementedError(f"Window type {window_type} not implemented!")


class NCSNPP_Wrapper(nn.Module):
    def __init__(
        self,
        n_fft=510,
        hop_length=128,
        num_frames=256,
        window="hann",
        spec_factor=0.15,
        spec_abs_exponent=0.5,
    ):
        super().__init__()
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

        self.net = NCSNpp(discriminative=True)

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

    def forward(self, batch_data):
        if "clean" in batch_data:
            x, y = batch_data["clean"], batch_data["perturbed"]

            current_len = x.size(-1)
            pad = max(self.target_len - current_len, 0)

            if pad == 0:
                # extract random part of the audio file
                start = int(np.random.uniform(0, current_len - self.target_len))
                x = x[..., start : start + self.target_len]
                y = y[..., start : start + self.target_len]
            else:
                # pad audio if the length T is smaller than num_frames
                x = F.pad(x, (pad // 2, pad // 2 + (pad % 2)), mode="constant")
                y = F.pad(y, (pad // 2, pad // 2 + (pad % 2)), mode="constant")

            Y = self.spec_fwd(self.stft(y)).unsqueeze(1)

            Y = self.net(Y)
            y_fake = self.istft(self.spec_back(Y.squeeze(1)), self.target_len)

            batch_data["clean"] = x
            batch_data["perturbed"] = y
            batch_data["fake"] = y_fake

        else:
            y = batch_data["perturbed"]
            T_orig = y.size(1)
            Y = self.spec_fwd(self.stft(y)).unsqueeze(1)
            Y = pad_spec(Y)
            Y = self.net(Y)
            y_denoised = self.istft(self.spec_back(Y.squeeze(1)), T_orig)
            batch_data["fake"] = y_denoised

        return batch_data
