import math
from functools import partial

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


def mag_phase(tensor):
    """Convert complex-valued tensor to magnitude and phase tensor."""
    return torch.sqrt(tensor[..., 0] ** 2 + tensor[..., 1] ** 2), torch.atan2(
        tensor[..., 1], tensor[..., 0]
    )


def mag_phase2(tensor):
    """Convert complex-valued tensor to magnitude and phase tensor."""
    mag = torch.sqrt(tensor[..., 0] ** 2 + tensor[..., 1] ** 2)
    phase = tensor / (mag.unsqueeze(-1) + 1e-9)
    return mag, phase


def stft_tranform(y, n_fft, hop_length, win_length, window_tensor, return_complex=True):
    """
    Args:
        y: [B, t]
        n_fft:
        hop_length:
        win_length:
        device:

    Returns:
        [B, F, T, 2]

    """
    assert y.dim() == 2
    features = torch.stft(
        y, n_fft, hop_length, win_length, window=window_tensor, return_complex=return_complex
    )
    return torch.view_as_real(features)


def istft_tranform(features, n_fft, hop_length, win_length, window_tensor, length=None):
    """Wrapper for the official torch.istft.

    Args:
        features: [B, F, T, 2] or [B, F, T] (complex)
        n_fft:
        hop_length:
        win_length:
        device:
        length:

    Returns:
        [B, t]
    """
    if features.dim() == 4:
        features = torch.view_as_complex(features)

    return torch.istft(
        features, n_fft, hop_length, win_length, window=window_tensor, length=length
    )


class STFTFeature(nn.Module):
    def __init__(
        self,
        n_fft=512,
        win_length=512,
        hop_length=128,
        window="hann",
        use_mag_phase=False,
        need_inverse=False,
        freq_high=None,
        sampling_rate=16000,
        compression=None,
        split_subbands=None,
        inverse_keys=["fake"],
    ):
        super().__init__()
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_fft = n_fft or 2 ** math.ceil(math.log2(self.win_length))
        windows = {
            "hann": torch.hann_window(self.win_length),
            "hamm": torch.hamming_window(self.win_length),
        }
        window_tensor = windows[window] if window in windows else None
        self.register_buffer("window", window_tensor)
        n_gpu = torch.cuda.device_count()
        if n_gpu > 0:
            if dist.is_initialized():
                self.rank = dist.get_rank()
            else:
                self.rank = 0
            self.window = self.window.to(f"cuda:{self.rank}")
        self.use_mag_phase = use_mag_phase
        self.stft = partial(
            stft_tranform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window_tensor=self.window,
        )
        self.istft = partial(
            istft_tranform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window_tensor=self.window,
        )
        self.need_inverse = need_inverse
        self.sampling_rate = sampling_rate
        self.high_cut_index = (
            int(freq_high / (self.sampling_rate) * self.n_fft + 0.5) if freq_high else None
        )
        self.compression = compression
        self.split_subbands = split_subbands
        self.inverse_keys = inverse_keys

    def forward(self, batch):
        # wavform: B, sig_len
        # STFT: B, F, T, 2
        perturbed = batch["perturbed"]
        perturbed_spec = self.stft(perturbed)
        if self.high_cut_index is not None:
            perturbed_spec[:, self.high_cut_index + 1 :] = 0
        if self.compression is not None:
            perturbed_mag, perturbed_phase = mag_phase2(perturbed_spec)
            if self.compression == "sqrt":
                perturbed_mag = perturbed_mag**0.5
            elif self.compression == "cubic":
                perturbed_mag = perturbed_mag**0.3
            elif self.compression == "log_1x":
                perturbed_mag = (perturbed_mag + 1.0).log()
            perturbed_spec = perturbed_mag.unsqueeze(-1) * perturbed_phase
        if not self.use_mag_phase:
            batch["perturbed_spectra"] = perturbed_spec
        else:
            perturbed_mag, perturbed_phase = mag_phase(perturbed_spec)
            batch["perturbed_mag"], batch["perturbed_phase"] = perturbed_mag, perturbed_phase
        perturbed_mag, _ = mag_phase(perturbed_spec)
        speech_mask = perturbed_mag.new_zeros(perturbed_mag.shape)
        spectra_length = batch["sample_length"].new_zeros(batch["sample_length"].shape)
        for i in range(perturbed_mag.shape[0]):
            cur_sample_length = batch["sample_length"][i]
            cur_sample_length = int(
                ((cur_sample_length + self.win_length) - self.win_length) / self.hop_length + 1
            )
            speech_mask[i, :, :cur_sample_length] = 1
            spectra_length[i] = cur_sample_length

        batch["speech_mask"] = speech_mask
        batch["spectra_length"] = spectra_length

        if self.split_subbands is not None:
            band_freq = self.n_fft // 2 // self.split_subbands
            perturbed_spec_subbands = []
            for i in range(self.split_subbands):
                perturbed_spec_subbands.append(
                    perturbed_spec[:, i * band_freq : (i + 1) * band_freq + 1]
                )
            perturbed_spec_subband = torch.stack(perturbed_spec_subbands, dim=1)
            batch["perturbed_subband_spectra"] = perturbed_spec_subband

        if "clean" in batch:
            clean = batch["clean"]
            clean_spec = self.stft(clean)
            if self.high_cut_index is not None:
                clean_spec[:, self.high_cut_index + 1 :] = 0
            if self.compression is not None:
                clean_mag, clean_phase = mag_phase2(clean_spec)
                if self.compression == "sqrt":
                    clean_mag = clean_mag**0.5
                elif self.compression == "cubic":
                    clean_mag = clean_mag**0.3
                elif self.compression == "log_1x":
                    clean_mag = (clean_mag + 1.0).log()
                clean_spec = clean_mag.unsqueeze(-1) * clean_phase
            if not self.use_mag_phase:
                batch["clean_spectra"] = clean_spec
            else:
                clean_mag, clean_phase = mag_phase(clean_spec)
                batch["clean_mag"], batch["clean_phase"] = clean_mag, clean_phase
            if self.split_subbands is not None:
                clean_spec_subbands = []
                for i in range(self.split_subbands):
                    clean_spec_subbands.append(
                        clean_spec[:, i * band_freq : (i + 1) * band_freq + 1]
                    )
                clean_spec_subband = torch.stack(clean_spec_subbands, dim=1)
                batch["clean_subband_spectra"] = clean_spec_subband

        return batch

    def inverse(self, batch):
        # STFT: B, F, T, 2
        # wavform: B, sig_len
        for inverse_key in self.inverse_keys:
            if self.split_subbands is not None:
                band_freq = self.n_fft // 2 // self.split_subbands
                enhan_subband_spec = batch[f"{inverse_key}_subband_spectra"]
                batch[f"{inverse_key}_spectra"] = torch.cat(
                    [
                        enhan_subband_spec[:, i, :band_freq]
                        if i < self.split_subbands - 1
                        else enhan_subband_spec[:, i]
                        for i in range(self.split_subbands)
                    ],
                    dim=1,
                )

            if not self.use_mag_phase:
                enhan_spec = batch[f"{inverse_key}_spectra"]
                if self.compression is not None:
                    enhan_mag, enhan_phase = mag_phase2(enhan_spec)
                    if self.compression == "sqrt":
                        enhan_mag = enhan_mag**2
                    elif self.compression == "cubic":
                        enhan_mag = enhan_mag ** (1 / 0.3)
                    elif self.compression == "log_1x":
                        enhan_mag = torch.exp(enhan_mag) - 1.0
                    enhan_spec = enhan_mag.unsqueeze(-1) * enhan_phase
                batch[f"{inverse_key}"] = self.istft(
                    enhan_spec, length=batch["perturbed"].shape[-1]
                )
            else:
                enhan_mag, enhan_phase = batch[f"{inverse_key}_mag"], batch[f"{inverse_key}_phase"]
                if self.compression is not None:
                    if self.compression == "sqrt":
                        enhan_mag = enhan_mag**2
                    elif self.compression == "cubic":
                        enhan_mag = enhan_mag ** (1 / 0.3)
                    elif self.compression == "log_1x":
                        enhan_mag = torch.exp(enhan_mag) - 1.0
                batch[f"{inverse_key}"] = self.istft(
                    (enhan_mag, enhan_phase), length=batch["perturbed"].shape[-1]
                )
        return batch
