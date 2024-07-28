"""Custom model losses."""

import torch
import torchaudio

SAMPLE_RATE = 48000
RESAMPLE_RATES = [8000, 16000, 24000]


class ContentCriteria(torch.nn.Module):
    """HiFi-GAN+ generator content losses.

    These are the non-adversarial content losses described in the original paper. The losses
    include L1 losses on the raw waveform, a set of log-scale STFTs, and the mel spectrogram.
    """

    def __init__(self) -> None:
        super().__init__()

        self._l1_loss = torch.nn.L1Loss()
        self._stft_xforms = torch.nn.ModuleList(
            [
                torchaudio.transforms.Spectrogram(
                    n_fft=frame_length,
                    hop_length=frame_length // 4,
                    power=1,
                )
                for frame_length in [512, 1024, 2048, 4096]
            ]
        )
        self._melspec_xform = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            f_min=RESAMPLE_RATES[0] // 2,
            f_max=SAMPLE_RATE // 2,
            n_fft=2048,
            win_length=int(0.025 * SAMPLE_RATE),
            hop_length=int(0.010 * SAMPLE_RATE),
            n_mels=128,
            power=1,
        )

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # L1 waveform loss
        wav_loss = self._l1_loss(y_pred, y_true)

        # L1 log spectrogram loss
        stft_loss = torch.tensor(0.0).to(y_pred.device)
        for stft in self._stft_xforms:
            s_true = torch.log(stft(y_true) + 1e-5)
            s_pred = torch.log(stft(y_pred) + 1e-5)
            stft_loss += self._l1_loss(s_pred, s_true)
        stft_loss /= len(self._stft_xforms)

        # mel spectrogram loss
        m_true = torch.log(self._melspec_xform(y_true) + 1e-5)
        m_pred = torch.log(self._melspec_xform(y_pred) + 1e-5)
        melspec_loss = self._l1_loss(m_pred, m_true)

        return wav_loss, stft_loss, melspec_loss


class HIFIGAN_adv_gen_loss(torch.nn.Module):
    """HiFi-GAN+ generator adversarial loss."""

    def __init__(self) -> None:
        super().__init__()
        self.gan_criteria = torch.nn.MSELoss()

    def forward(self, y_fake) -> torch.Tensor:
        return self.gan_criteria(y_fake, torch.ones_like(y_fake))


class HIFIGAN_feat_loss(torch.nn.Module):
    """HiFi-GAN+ feature matching loss."""

    def __init__(self) -> None:
        super().__init__()
        self.feat_criteria = torch.nn.L1Loss()

    def forward(self, f_dsc_fake, f_dsc_real) -> torch.Tensor:
        feat_loss = sum(
            (self.feat_criteria(fake, real) for (fake, real) in zip(f_dsc_fake, f_dsc_real)),
            start=torch.tensor(0),
        ) / len(f_dsc_fake)
        return feat_loss


class HIFIGAN_adv_dsc_loss(torch.nn.Module):
    """HiFi-GAN+ discriminator adversarial loss."""

    def __init__(self) -> None:
        super().__init__()
        self.gan_criteria = torch.nn.MSELoss()

    def forward(self, y_real, y_fake) -> torch.Tensor:
        return (
            self.gan_criteria(y_real, torch.ones_like(y_real))
            + self.gan_criteria(y_fake, torch.zeros_like(y_fake))
        ) / 2
