import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from src.models.components.GAN.discriminator.hifigan_vocoder.hifigan_dicriminator import (
    hifigan_vocoder_discriminator_adv_dsc_loss,
    hifigan_vocoder_discriminator_adv_gen_loss,
    hifigan_vocoder_discriminator_feat_loss,
)


class LSGAN_G_Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, batch_data):
        loss = 0
        for logits in batch_data["predicted_fake_logits"]:
            for layer_logits in logits:
                loss += F.mse_loss(layer_logits, torch.ones_like(layer_logits))
        batch_data["loss_G"] = loss
        return batch_data


class LSGAN_D_Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, batch_data):
        loss = 0
        for fake_logits, real_logits in zip(
            batch_data["predicted_fake_logits"], batch_data["predicted_clean_logits"]
        ):
            for i in range(len(fake_logits)):
                loss += F.mse_loss(fake_logits[i], torch.zeros_like(fake_logits[i])) + F.mse_loss(
                    real_logits[i], torch.ones_like(real_logits[i])
                )
        batch_data["loss_D"] = loss
        return batch_data


class HIFIGAN_Vocoder_D_Loss(nn.Module):
    def __init__(self, enhanced_key="fake"):
        super().__init__()
        self.adv_dsc_loss = hifigan_vocoder_discriminator_adv_dsc_loss()
        self.enhanced_key = enhanced_key

    def forward(self, batch_data):
        real_logits = batch_data["predicted_clean_logits"]
        fake_logits = batch_data[f"predicted_{self.enhanced_key}_logits"]
        adv_dsc_loss = self.adv_dsc_loss(real_logits, fake_logits)
        batch_data["loss_D_adv_dsc"] = adv_dsc_loss
        batch_data["loss_D"] = adv_dsc_loss
        return batch_data


class WavSpecConvergence_Loss(nn.Module):
    def __init__(
        self,
        sampling_rate=48000,
        alpha_wav_l1=1.0,
        alpha_mag_l2=1.0,
        alpha_mag_log=1.0,
        alpha_mag_norm_l2=1.0,
        alpha_mel_log=1.0,
        alpha_mel_l2=1.0,
        enhanced_key="fake",
        loss_prefix="loss_G",
    ):
        super().__init__()
        DEFAULT_SAMPLING_RATE = 48000
        sample_rate_ratio = sampling_rate / DEFAULT_SAMPLING_RATE
        self._l1_loss = torch.nn.L1Loss()
        self._l2_loss = torch.nn.MSELoss()
        self._stft_xforms = torch.nn.ModuleList(
            [
                torchaudio.transforms.Spectrogram(
                    n_fft=frame_length,
                    hop_length=frame_length // 4,
                    power=1,
                )
                for frame_length in [int(512 * sample_rate_ratio), int(1024 * sample_rate_ratio), int(2048 * sample_rate_ratio), int(4096 * sample_rate_ratio)]
            ]
        )
        self._melspec_xform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sampling_rate,
            f_min=0,
            f_max=sampling_rate // 2,
            n_fft=2048,
            win_length=int(0.025 * sampling_rate),
            hop_length=int(0.010 * sampling_rate),
            n_mels=128,
            power=1,
        )

        # n_gpu = torch.cuda.device_count()
        # if n_gpu > 1:
        #     self.rank = dist.get_rank()
        # if n_gpu > 1:
        #     self._stft_xforms = self._stft_xforms.to(f"cuda:{self.rank}")
        #     self._melspec_xform = self._melspec_xform.to(f"cuda:{self.rank}")
        # elif n_gpu > 0:
        #     self._stft_xforms = self._stft_xforms.to("cuda")
        #     self._melspec_xform = self._melspec_xform.to("cuda")

        self.alpha_wav_l1 = alpha_wav_l1
        self.alpha_mag_l2 = alpha_mag_l2
        self.alpha_mag_log = alpha_mag_log
        self.alpha_mag_norm_l2 = alpha_mag_norm_l2
        self.alpha_mel_log = alpha_mel_log
        self.alpha_mel_l2 = alpha_mel_l2
        self.enhanced_key = enhanced_key
        self.loss_prefix = loss_prefix

    def calc_convergence_loss(self, clean, enhanced):
        # wavform loss
        wav_l1_loss = self._l1_loss(enhanced, clean)
        # STFT loss
        mag_l2_loss = torch.tensor(0.0).to(enhanced.device)
        mag_log_loss = torch.tensor(0.0).to(enhanced.device)
        mag_norm_l2_loss = torch.tensor(0.0).to(enhanced.device)
        for stft in self._stft_xforms:
            mag_enhanced = stft(enhanced)
            mag_clean = stft(clean)
            mag_l2_loss += self._l2_loss(mag_enhanced, mag_clean)
            mag_log_loss += self._l1_loss(
                torch.log(mag_enhanced * 32768 + 1e-6), torch.log(mag_clean * 32768 + 1e-6)
            )
            mag_norm_l2_loss += (
                ((mag_clean - mag_enhanced) ** 2).sum(-1).sum(-1).sqrt()
                / ((mag_clean**2).sum(-1).sum(-1).sqrt() + 1e-6)
            ).mean()
        mag_log_loss /= len(self._stft_xforms)
        mag_norm_l2_loss /= len(self._stft_xforms)
        # MelSpectrogram loss
        mel_enhanced = self._melspec_xform(enhanced)
        mel_clean = self._melspec_xform(clean)
        mel_log_loss = self._l1_loss(
            torch.log(mel_enhanced * 32768 + 1e-6), torch.log(mel_clean * 32768 + 1e-6)
        )
        mel_l2_loss = self._l2_loss(mel_enhanced, mel_clean)
        wav_l1_loss = self.alpha_wav_l1 * wav_l1_loss
        mag_l2_loss = self.alpha_mag_l2 * mag_l2_loss
        mag_log_loss = self.alpha_mag_log * mag_log_loss
        mag_norm_l2_loss = self.alpha_mag_norm_l2 * mag_norm_l2_loss
        mel_log_loss = self.alpha_mel_log * mel_log_loss
        mel_l2_loss = self.alpha_mel_l2 * mel_l2_loss
        # import ipdb; ipdb.set_trace()
        return wav_l1_loss, mag_l2_loss, mag_log_loss, mag_norm_l2_loss, mel_log_loss, mel_l2_loss

    def forward(self, batch_data):
        clean = batch_data["clean"]
        enhanced = batch_data[self.enhanced_key]
        (
            wav_l1_loss,
            mag_l2_loss,
            mag_log_loss,
            mag_norm_l2_loss,
            mel_log_loss,
            mel_l2_loss,
        ) = self.calc_convergence_loss(clean, enhanced)
        batch_data[f"{self.loss_prefix}_wav_l1"] = wav_l1_loss
        batch_data[f"{self.loss_prefix}_mag_l2"] = mag_l2_loss
        batch_data[f"{self.loss_prefix}_mag_log"] = mag_log_loss
        batch_data[f"{self.loss_prefix}_mag_norm_l2"] = mag_norm_l2_loss
        batch_data[f"{self.loss_prefix}_mel_log"] = mel_log_loss
        batch_data[f"{self.loss_prefix}_mel_l2"] = mel_l2_loss
        batch_data[f"{self.loss_prefix}"] = (
            wav_l1_loss
            + mag_l2_loss
            + mag_log_loss
            + mag_norm_l2_loss
            + mel_log_loss
            + mel_l2_loss
        )
        return batch_data


class WavSpecConvergence_HIFIGAN_Vocoder_G_Loss(nn.Module):
    def __init__(
        self,
        sampling_rate=48000,
        alpha_wav_l1=1.0,
        alpha_mag_l2=1.0,
        alpha_mag_log=1.0,
        alpha_mag_norm_l2=1.0,
        alpha_mel_log=1.0,
        alpha_mel_l2=1.0,
        alpha_adv_gen=1.0,
        alpha_adv_feat=1.0,
        enhanced_key="fake",
    ):
        super().__init__()
        DEFAULT_SAMPLING_RATE = 48000
        sample_rate_ratio = sampling_rate / DEFAULT_SAMPLING_RATE
        self._l1_loss = torch.nn.L1Loss()
        self._l2_loss = torch.nn.MSELoss()
        self._stft_xforms = torch.nn.ModuleList(
            [
                torchaudio.transforms.Spectrogram(
                    n_fft=frame_length,
                    hop_length=frame_length // 4,
                    power=1,
                )
                for frame_length in [int(512 * sample_rate_ratio), int(1024 * sample_rate_ratio), int(2048 * sample_rate_ratio), int(4096 * sample_rate_ratio)]
            ]
        )
        self._melspec_xform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sampling_rate,
            f_min=0,
            f_max=sampling_rate // 2,
            n_fft=2048,
            win_length=int(0.025 * sampling_rate),
            hop_length=int(0.010 * sampling_rate),
            n_mels=128,
            power=1,
        )

        # n_gpu = torch.cuda.device_count()
        # if n_gpu > 1:
        #     self.rank = dist.get_rank()
        # if n_gpu > 1:
        #     self._stft_xforms = self._stft_xforms.to(f"cuda:{self.rank}")
        #     self._melspec_xform = self._melspec_xform.to(f"cuda:{self.rank}")
        # elif n_gpu > 0:
        #     self._stft_xforms = self._stft_xforms.to("cuda")
        #     self._melspec_xform = self._melspec_xform.to("cuda")

        self.adv_gen_loss = hifigan_vocoder_discriminator_adv_gen_loss()
        self.adv_feat_loss = hifigan_vocoder_discriminator_feat_loss()
        self.alpha_wav_l1 = alpha_wav_l1
        self.alpha_mag_l2 = alpha_mag_l2
        self.alpha_mag_log = alpha_mag_log
        self.alpha_mag_norm_l2 = alpha_mag_norm_l2
        self.alpha_mel_log = alpha_mel_log
        self.alpha_mel_l2 = alpha_mel_l2
        self.alpha_adv_gen = alpha_adv_gen
        self.alpha_adv_feat = alpha_adv_feat
        self.enhanced_key = enhanced_key

    def calc_convergence_loss(self, clean, enhanced):
        # wavform loss
        wav_l1_loss = self._l1_loss(enhanced, clean)
        # STFT loss
        mag_l2_loss = torch.tensor(0.0).to(enhanced.device)
        mag_log_loss = torch.tensor(0.0).to(enhanced.device)
        mag_norm_l2_loss = torch.tensor(0.0).to(enhanced.device)
        for stft in self._stft_xforms:
            mag_enhanced = stft(enhanced)
            mag_clean = stft(clean)
            mag_l2_loss += self._l2_loss(mag_enhanced, mag_clean)
            mag_log_loss += self._l1_loss(
                torch.log(mag_enhanced * 32768 + 1e-6), torch.log(mag_clean * 32768 + 1e-6)
            )
            mag_norm_l2_loss += (
                ((mag_clean - mag_enhanced) ** 2).sum(-1).sum(-1).sqrt()
                / ((mag_clean**2).sum(-1).sum(-1).sqrt() + 1e-6)
            ).mean()
        mag_log_loss /= len(self._stft_xforms)
        mag_norm_l2_loss /= len(self._stft_xforms)
        # MelSpectrogram loss
        mel_enhanced = self._melspec_xform(enhanced)
        mel_clean = self._melspec_xform(clean)
        mel_log_loss = self._l1_loss(
            torch.log(mel_enhanced * 32768 + 1e-6), torch.log(mel_clean * 32768 + 1e-6)
        )
        mel_l2_loss = self._l2_loss(mel_enhanced, mel_clean)
        wav_l1_loss = self.alpha_wav_l1 * wav_l1_loss
        mag_l2_loss = self.alpha_mag_l2 * mag_l2_loss
        mag_log_loss = self.alpha_mag_log * mag_log_loss
        mag_norm_l2_loss = self.alpha_mag_norm_l2 * mag_norm_l2_loss
        mel_log_loss = self.alpha_mel_log * mel_log_loss
        mel_l2_loss = self.alpha_mel_l2 * mel_l2_loss
        # import ipdb; ipdb.set_trace()
        return wav_l1_loss, mag_l2_loss, mag_log_loss, mag_norm_l2_loss, mel_log_loss, mel_l2_loss

    def calc_adv_gen_loss(self, fake_logits, real_feat_list, fake_feat_list):
        adv_gen_loss = self.adv_gen_loss(fake_logits)
        adv_feat_loss = self.adv_feat_loss(real_feat_list, fake_feat_list)
        adv_gen_loss = self.alpha_adv_gen * adv_gen_loss
        adv_feat_loss = self.alpha_adv_feat * adv_feat_loss
        return adv_gen_loss, adv_feat_loss

    def forward(self, batch_data):
        clean = batch_data["clean"]
        fake_logits = batch_data[f"predicted_{self.enhanced_key}_logits"]
        real_feat_list = batch_data["predicted_clean_feature_list"]
        fake_feat_list = batch_data[f"predicted_{self.enhanced_key}_feature_list"]
        enhanced = batch_data[self.enhanced_key]
        (
            wav_l1_loss,
            mag_l2_loss,
            mag_log_loss,
            mag_norm_l2_loss,
            mel_log_loss,
            mel_l2_loss,
        ) = self.calc_convergence_loss(clean, enhanced)
        adv_gen_loss, adv_feat_loss = self.calc_adv_gen_loss(
            fake_logits, real_feat_list, fake_feat_list
        )
        batch_data["loss_G_wav_l1"] = wav_l1_loss
        batch_data["loss_G_mag_l2"] = mag_l2_loss
        batch_data["loss_G_mag_log"] = mag_log_loss
        batch_data["loss_G_mag_norm_l2"] = mag_norm_l2_loss
        batch_data["loss_G_mel_log"] = mel_log_loss
        batch_data["loss_G_mel_l2"] = mel_l2_loss
        batch_data["loss_G_adv_gen"] = adv_gen_loss
        batch_data["loss_G_adv_feat"] = adv_feat_loss
        batch_data["loss_G"] = (
            wav_l1_loss
            + mag_l2_loss
            + mag_log_loss
            + mag_norm_l2_loss
            + mel_log_loss
            + mel_l2_loss
            + adv_gen_loss
            + adv_feat_loss
        )
        return batch_data
