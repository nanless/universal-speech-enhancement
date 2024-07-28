import torch
import torch.nn as nn
import torchaudio

from ..hifigan.open_models import WaveDiscriminator
from .hifigan import MultiPeriodDiscriminator, MultiScaleDiscriminator

SAMPLE_RATE = 24000


class MelspecDiscriminator(torch.nn.Module):
    """Mel spectrogram (frequency domain) discriminator."""

    def __init__(self, n_fft, win_length, hop_length, n_mels) -> None:
        super().__init__()

        # mel filterbank transform
        self._melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            power=1,
        )

        # time-frequency 2D convolutions
        kernel_sizes = [(7, 7), (4, 4), (4, 4), (4, 4)]
        strides = [(1, 2), (1, 2), (1, 2), (1, 2)]
        self._convs = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    torch.nn.Conv2d(
                        in_channels=1 if i == 0 else 32,
                        out_channels=64,
                        kernel_size=k,
                        stride=s,
                        padding=(1, 2),
                        bias=False,
                    ),
                    torch.nn.InstanceNorm2d(num_features=64),
                    torch.nn.GLU(dim=1),
                )
                for i, (k, s) in enumerate(zip(kernel_sizes, strides))
            ]
        )

        # output adversarial projection
        self._postnet = torch.nn.Conv2d(
            in_channels=32,
            out_channels=1,
            kernel_size=(15, 5),
            stride=(1, 2),
        )

    def forward(self, x: torch.Tensor):
        # apply the log-scale mel spectrogram transform
        x = torch.log(self._melspec(x) + 1e-5)

        # compute hidden layers and feature maps
        f = []
        for c in self._convs:
            x = c(x)
            f.append(x)

        # apply the output projection and global average pooling
        x = self._postnet(x)
        x = x.mean(dim=[-2, -1])

        return x, f


class MultiMelSpecDiscriminator(torch.nn.Module):
    def __init__(
        self,
        n_ffts=[1024, 2048, 512],
        win_lengths=[960, 1920, 480],
        hop_lengths=[240, 480, 120],
        n_mels=[200, 256, 128],
    ):
        super().__init__()
        self._discriminators = torch.nn.ModuleList(
            [
                MelspecDiscriminator(n_fft, win_length, hop_length, n_mel)
                for n_fft, win_length, hop_length, n_mel in zip(
                    n_ffts, win_lengths, hop_lengths, n_mels
                )
            ]
        )

    def forward(self, x):
        y_d_rs = []
        fmap_rs = []
        for d in self._discriminators:
            y_d, fmap = d(x)
            y_d_rs.append(y_d)
            fmap_rs.append(fmap)
        return y_d_rs, fmap_rs


class MultiWaveDiscriminator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._discriminators = torch.nn.ModuleList(
            [
                WaveDiscriminator(sample_rate=8000),
                WaveDiscriminator(sample_rate=12000),
                WaveDiscriminator(sample_rate=16000),
                WaveDiscriminator(sample_rate=24000),
            ]
        )

    def forward(self, x):
        f = []
        y = []
        for dsc_model in self._discriminators:
            yi, fi = dsc_model(x)
            y.append(yi)
            f.extend(fi)
        return y, f


class hifigan_vocoder_discriminator_24k(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.MPD = MultiPeriodDiscriminator(
            periods=[2, 3, 5, 7, 11],
            discriminator_params={
                "in_channels": 1,
                "out_channels": 1,
                "kernel_sizes": [5, 3],
                "channels": 32,
                "downsample_scales": [3, 3, 3, 3, 1],
                "max_downsample_channels": 1024,
                "bias": True,
                "nonlinear_activation": "LeakyReLU",
                "nonlinear_activation_params": {"negative_slope": 0.1},
                "use_spectral_norm": False,
            },
        )
        self.MSD = MultiScaleDiscriminator(
            scales=3,
            downsample_pooling="DWT",
            downsample_pooling_params={
                "kernel_size": 4,
                "stride": 2,
                "padding": 2,
            },
            discriminator_params={
                "in_channels": 1,
                "out_channels": 1,
                "kernel_sizes": [15, 41, 5, 3],
                "channels": 128,
                "max_downsample_channels": 1024,
                "max_groups": 16,
                "bias": True,
                "downsample_scales": [4, 4, 4, 4, 1],
                "nonlinear_activation": "LeakyReLU",
                "nonlinear_activation_params": {"negative_slope": 0.1},
            },
            follow_official_norm=True,
        )
        self.MMD = MultiMelSpecDiscriminator(
            n_ffts=[1024, 256, 512],
            win_lengths=[960, 240, 480],
            hop_lengths=[240, 60, 120],
            n_mels=[128, 64, 80],
        )

    def forward_fake(self, batch_data):
        x = batch_data["enhanced"].unsqueeze(1)
        logits_MPD, feature_list_MPD = self.MPD(x)
        logits_MSD, feature_list_MSD = self.MSD(x)
        logits_MMD, feature_list_MMD = self.MMD(x)

        logits = [logits_MPD, logits_MSD, logits_MMD]
        feature_list = [feature_list_MPD, feature_list_MSD, feature_list_MMD]
        batch_data["predicted_enhanced_logits"] = logits
        batch_data["predicted_enhanced_feature_list"] = feature_list
        return batch_data

    def forward_real(self, batch_data):
        x = batch_data["clean"].unsqueeze(1)
        logits_MPD, feature_list_MPD = self.MPD(x)
        logits_MSD, feature_list_MSD = self.MSD(x)
        logits_MMD, feature_list_MMD = self.MMD(x)
        logits = [logits_MPD, logits_MSD, logits_MMD]
        feature_list = [feature_list_MPD, feature_list_MSD, feature_list_MMD]
        batch_data["predicted_clean_logits"] = logits
        batch_data["predicted_clean_feature_list"] = feature_list
        return batch_data

    def forward(self, batch_data):
        batch_data = self.forward_fake(batch_data)
        batch_data = self.forward_real(batch_data)
        return batch_data


class hifigan_vocoder_discriminator_24k_MVD(nn.Module):
    def __init__(self, enhanced_key="enhanced"):
        super().__init__()
        self.MPD = MultiPeriodDiscriminator(
            periods=[2, 3, 5, 7, 11],
            discriminator_params={
                "in_channels": 1,
                "out_channels": 1,
                "kernel_sizes": [5, 3],
                "channels": 32,
                "downsample_scales": [3, 3, 3, 3, 1],
                "max_downsample_channels": 1024,
                "bias": True,
                "nonlinear_activation": "LeakyReLU",
                "nonlinear_activation_params": {"negative_slope": 0.1},
                "use_spectral_norm": False,
            },
        )
        self.MVD = MultiWaveDiscriminator()
        self.MMD = MultiMelSpecDiscriminator(
            n_ffts=[1024, 256, 512],
            win_lengths=[960, 240, 480],
            hop_lengths=[240, 60, 120],
            n_mels=[128, 64, 80],
        )
        self.enhanced_key = enhanced_key

    def forward_fake(self, batch_data):
        x = batch_data[f"{self.enhanced_key}"].unsqueeze(1)
        logits_MPD, feature_list_MPD = self.MPD(x)
        logits_MVD, feature_list_MVD = self.MVD(x)
        logits_MMD, feature_list_MMD = self.MMD(x)

        logits = [logits_MPD, logits_MVD, logits_MMD]
        feature_list = [feature_list_MPD, feature_list_MVD, feature_list_MMD]
        batch_data[f"predicted_{self.enhanced_key}_logits"] = logits
        batch_data[f"predicted_{self.enhanced_key}_feature_list"] = feature_list
        return batch_data

    def forward_real(self, batch_data):
        x = batch_data["clean"].unsqueeze(1)
        logits_MPD, feature_list_MPD = self.MPD(x)
        logits_MVD, feature_list_MVD = self.MVD(x)
        logits_MMD, feature_list_MMD = self.MMD(x)
        logits = [logits_MPD, logits_MVD, logits_MMD]
        feature_list = [feature_list_MPD, feature_list_MVD, feature_list_MMD]
        batch_data["predicted_clean_logits"] = logits
        batch_data["predicted_clean_feature_list"] = feature_list
        return batch_data

    def forward(self, batch_data):
        batch_data = self.forward_fake(batch_data)
        batch_data = self.forward_real(batch_data)
        return batch_data


class hifigan_vocoder_discriminator_adv_gen_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.gan_criteria = torch.nn.MSELoss()

    def forward(self, y_fake) -> torch.Tensor:
        loss_total = 0
        cnt_total = 0
        for type_i in range(len(y_fake)):
            for dsc_i in range(len(y_fake[type_i])):
                loss_total += self.gan_criteria(
                    y_fake[type_i][dsc_i], torch.ones_like(y_fake[type_i][dsc_i])
                )
                cnt_total += 1
        loss_total /= cnt_total
        return loss_total


class hifigan_vocoder_discriminator_feat_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.feat_criteria = torch.nn.L1Loss()

    def forward(self, f_dsc_real, f_dsc_fake) -> torch.Tensor:
        loss_total = 0
        cnt_total = 0
        for type_i in range(len(f_dsc_fake)):
            for dsc_i in range(len(f_dsc_fake[type_i])):
                for layer_i in range(len(f_dsc_fake[type_i][dsc_i])):
                    loss_total += self.feat_criteria(
                        f_dsc_fake[type_i][dsc_i][layer_i], f_dsc_real[type_i][dsc_i][layer_i]
                    )
                    cnt_total += 1
        loss_total /= cnt_total
        return loss_total


class hifigan_vocoder_discriminator_adv_dsc_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.gan_criteria = torch.nn.MSELoss()

    def forward(self, y_real, y_fake) -> torch.Tensor:
        loss_total = 0
        cnt_total = 0
        for type_i in range(len(y_real)):
            for dsc_i in range(len(y_real[type_i])):
                loss_total += self.gan_criteria(
                    y_real[type_i][dsc_i], torch.ones_like(y_real[type_i][dsc_i])
                )
                loss_total += self.gan_criteria(
                    y_fake[type_i][dsc_i], torch.zeros_like(y_fake[type_i][dsc_i])
                )
                cnt_total += 2
        loss_total /= cnt_total
        return loss_total
