import gc
import json
import pickle
from collections import OrderedDict

import librosa
import numpy as np
import soundfile as sf
import torch.distributed as dist
from memory_profiler import profile
from scipy.signal import fftconvolve
from torch.utils.data import Dataset

from .FRA_RIR import FRA_RIR
from .perturb import (
    AACConversionPerturb,
    BandRejectPerturb,
    BassBoostPerturb,
    BitCrushPerturb,
    ColoredNoisePerturb,
    DCOffsetPerturb,
    DRCPerturb,
    EQMuchGainPerturb,
    EQPerturb,
    GSMcodecsPerturb,
    LoudnessPerturb,
    LowPassPerturb,
    MP3CompressorPerturb,
    OPUSCodecsPerturb,
    PacketLossPerturb,
    PitchPerturb,
    SpeakerDistortionPerturbHardClip,
    SpeakerDistortionPerturbHardClipOnRate,
    SpeakerDistortionPerturbPedal,
    SpeakerDistortionPerturbSigmoid1,
    SpeakerDistortionPerturbSigmoid2,
    SpeakerDistortionPerturbSoftClip,
    SpeakerDistortionPerturbSox,
    SpectralLeakagePerturb,
    SpectralTimeFreqHolesPerturb,
    SpeedPerturb,
)
from .webrtc_utils import WebRTCNS_perturb, WebRTCSAGC_perturb


class Dataset(Dataset):
    def __init__(
        self,
        # clean and noise path
        clean_list_path=None,
        clean_json_path=None,
        noise_list_path=None,
        noise_json_path=None,
        check_list_files=True,
        # number of speakers
        min_n_speakers=1,
        max_n_speakers=3,
        # whether use duration filter
        min_duration_seconds=None,
        max_duration_seconds=None,
        # whether to remove DC offset
        remove_dc_offset=False,
        # target sampling rate
        sampling_rate=None,
        resample_method="soxr_vhq",
        # speech splice configuration
        speech_splice=False,
        speech_splice_equal_volume=False,
        speech_splice_equal_volume_range=[-6, 6],
        speech_splice_seconds=10,
        speech_random_start=False,
        add_extra_space_prob=0,
        # reverbrate configuration
        reverb_prob=0,
        reverb_use_FRA=False,
        rir_list_path=None,
        reverb_noise=False,
        min_rt60=None,
        max_rt60=None,
        # add noise configuration
        add_noise_prob=0,
        only_noise_prob=0,
        noise_repeat_splice=False,
        trim_noise=False,
        snr_min=None,
        snr_max=None,
        noise_mix_prob=0,
        # speed perturb configuration
        speed_perturb_prob=0,
        speed_rate_min=0.8,
        speed_rate_max=1.2,
        # pitch perturb configuration
        pitch_shift_prob=0,
        semitones_down=-1.5,
        semitones_up=1.5,
        # loudness perturb configuration
        loudness_perturb_prob=0,
        loudness_min_factor=0.1,
        loudness_max_factor=10,
        loudness_max_n_intervals=5,
        # hard clipping perturb configuration
        clip_prob=0,
        hard_clip_portion=0.5,
        hard_clip_on_rate=True,
        hard_clip_rate_min=0.01,
        hard_clip_rate_max=0.2,
        hard_clip_threshold_db_min=-40,
        hard_clip_threshold_db_max=0,
        # sorf clipping perturb configuration
        soft_clip_types=["sox", "pedal", "soft", "sigmoid1", "sigmoid2"],
        # eq perturb configuration
        eq_perturb_prob=0,
        eq_db_min=-5,
        eq_db_max=0,
        # eq much gain perturb configuration
        eq_much_gain_prob=0,
        eq_much_gain_db_min=5,
        eq_much_gain_db_max=25,
        eq_much_gain_freq_min=1000,
        eq_much_gain_freq_max=16000,
        # band reject perturb configuration
        band_reject_prob=0,
        band_reject_min_center_freq=100,
        band_reject_max_center_freq=22000,
        band_reject_min_q=1,
        band_reject_max_q=8,
        band_reject_min_freq_bandwidth=100,
        band_reject_max_freq_bandwidth=2000,
        band_reject_use_stft=False,
        band_reject_max_n=2,
        # bass effect
        bass_boost_prob=0,
        bass_boost_highpass_cutoff_min=500,
        bass_boost_highpass_cutoff_max=2000,
        bass_boost_attenuation_min_db=-20,
        # DC offset perturb configuration
        dc_offset_prob=0,
        dc_offset_min=0.001,
        dc_offset_max=0.2,
        # spectral leakage perturb configuration
        spectral_leakage_prob=0,
        spectral_leakage_window_lengths=[1024, 2048, 4096],
        spectral_leakage_max_time_shift=20,
        # colored noise perturb configuration
        colored_noise_prob=0,
        colered_noise_snr_min=5,
        colered_noise_snr_max=50,
        colered_noise_types=["white", "pink", "brown", "equalized"],
        # low pass perturb configuration
        lowpass_prob=0,
        lowpass_min_cutoff_freq=1000,
        lowpass_max_cutoff_freq=24000,
        lowpass_min_order=4,
        lowpass_max_order=20,
        # spectral time frequency holes perturb configuration
        spectral_time_freq_holes_prob=0,
        spectral_time_freq_holes_stft_frame_length=1024,
        spectral_time_freq_holes_stft_frame_step=256,
        spectral_time_freq_holes_stft_holes_num_min=1,
        spectral_time_freq_holes_stft_holes_num_max=20,
        spectral_time_freq_holes_stft_holes_width_min_freq=1,
        spectral_time_freq_holes_stft_holes_width_max_freq=5,
        spectral_time_freq_holes_stft_holes_width_min_time=1,
        spectral_time_freq_holes_stft_holes_width_max_time=5,
        spectral_time_freq_holes_cutoff_freq=10000,
        # webrtc ns configuration
        webrtc_ns_prob=0,
        webrtc_ns_levels=[0, 1, 2, 3],
        webrtc_ns_volume_protection=True,
        # webrtc agc configuration
        webrtc_agc_prob=0,
        webrtc_agc_target_level_dbfs_max=0,
        webrtc_agc_target_level_dbfs_min=-31,
        # drc configuration
        drc_prob=0,
        drc_threshold_db_min=-50,
        drc_threshold_db_max=0,
        drc_ratio_min=1,
        drc_ratio_max=20,
        drc_attack_ms_min=0.5,
        drc_attack_ms_max=5.0,
        drc_release_ms_min=50,
        drc_release_ms_max=1000,
        # codecs perturb configuration
        codecs_prob=0,
        codecs_types=["mp3", "aac", "gsm", "opus"],
        # packet loss perturb configuration
        packet_loss_prob=0,
        packet_loss_rate_min=0,
        packet_loss_rate_max=0.3,
        packet_loss_frame_time_min=0.008,
        packet_loss_frame_time_max=0.05,
        packet_loss_decay_rate_min=0,
        packet_loss_decay_rate_max=0.2,
        packet_loss_hard_loss_prob=1.0,
        packet_loss_on_vad=False,
        # bit crush perturb configuration
        bit_crush_prob=0,
        bit_crush_bit_min=4,
        bit_crush_bit_max=32,
        # colored noise post perturb configuration
        colored_noise_post_prob=0,
        colored_noise_post_snr_min=5,
        colored_noise_post_snr_max=50,
        colored_noise_post_types=["white", "pink", "brown"],
        # random volume configuration
        random_volume=False,
        volume_min_dB=None,
        volume_max_dB=None,
        volume_min_sample=None,
        volume_max_sample=None,
        use_rms_volume=False,
        sync_random_volume=False,
        # output cut configuration
        output_cut_seconds=None,
        output_random_cut=False,
        output_normalize=False,
        output_resample=False,
        output_resample_rate=None,
        # debug flag
        debug=False,
        dummy=False,
    ):
        super().__init__()
        assert sampling_rate is not None
        self.min_n_speakers = min_n_speakers
        self.max_n_speakers = max_n_speakers
        self.sampling_rate = sampling_rate
        self.resample_method = resample_method
        self.min_duration_seconds = min_duration_seconds
        self.max_duration_seconds = max_duration_seconds
        self.remove_dc_offset = remove_dc_offset
        self.check_list_files = check_list_files

        if dist.is_initialized():
            self.rank = dist.get_rank()
        else:
            self.rank = 0

        if clean_json_path and noise_json_path:
            self.clean_list, self.clean_duration_list, self.noise_list = self.parse_json_double(
                clean_json_path, noise_json_path
            )
        else:
            self.clean_list, self.clean_duration_list, self.noise_list = self.parse_list(
                clean_list_path, noise_list_path
            )

        # reverberate
        self.reverb_prob = reverb_prob
        self.add_reverb = bool(self.reverb_prob > 0)
        self.reverb_use_FRA = reverb_use_FRA
        if self.add_reverb:
            if not self.reverb_use_FRA:
                self.rir_list = [line.rstrip("\n") for line in open(rir_list_path)]
        self.reverb_noise = reverb_noise
        self.min_rt60 = min_rt60
        self.max_rt60 = max_rt60

        # augmentation
        self.speed_perturb_prob = speed_perturb_prob
        if self.speed_perturb_prob > 0:
            self.speed_perturber = SpeedPerturb(
                sample_rate=self.sampling_rate,
                min_speed_rate=speed_rate_min,
                max_speed_rate=speed_rate_max,
            )
        self.pitch_shift_prob = pitch_shift_prob
        if self.pitch_shift_prob > 0:
            self.pitch_shifter = PitchPerturb(
                sample_rate=self.sampling_rate,
                down_max_semitone=semitones_down,
                up_max_semitone=semitones_up,
            )

        # splice input speech
        self.speech_splice = speech_splice
        self.speech_splice_equal_volume = speech_splice_equal_volume
        self.speech_splice_equal_volume_range = speech_splice_equal_volume_range
        self.speech_splice_seconds = speech_splice_seconds
        self.speech_splice_length = int(speech_splice_seconds * sampling_rate)
        self.speech_random_start = speech_random_start
        self.add_extra_space_prob = add_extra_space_prob

        # snr range
        assert add_noise_prob is not None
        if add_noise_prob:
            assert snr_min is not None
            assert snr_max is not None
        self.add_noise_prob = add_noise_prob
        self.only_noise_prob = only_noise_prob
        self.snr_min = snr_min
        self.snr_max = snr_max
        self.noise_repeat_splice = noise_repeat_splice
        self.trim_noise = trim_noise
        self.noise_mix_prob = noise_mix_prob

        # loudness perturb
        self.loudness_perturb_prob = loudness_perturb_prob
        if self.loudness_perturb_prob > 0:
            self.loudness_perturber = LoudnessPerturb(
                sample_rate=self.sampling_rate,
                min_factor=loudness_min_factor,
                max_factor=loudness_max_factor,
                max_n_intervals=loudness_max_n_intervals,
            )

        # clipping perturb
        self.clip_prob = clip_prob
        self.hard_clip_portion = hard_clip_portion
        if self.clip_prob > 0 and self.hard_clip_portion > 0:
            if hard_clip_on_rate:
                self.hard_clip_perturber = SpeakerDistortionPerturbHardClipOnRate(
                    sample_rate=self.sampling_rate,
                    clip_rate_min=hard_clip_rate_min,
                    clip_rate_max=hard_clip_rate_max,
                )
            else:
                self.hard_clip_perturber = SpeakerDistortionPerturbHardClip(
                    sample_rate=self.sampling_rate,
                    threshold_db_min=hard_clip_threshold_db_min,
                    threshold_db_max=hard_clip_threshold_db_max,
                )
        if self.clip_prob > 0 and self.hard_clip_portion < 1:
            self.soft_clip_types = soft_clip_types
            self.soft_clip_perturbers = list()
            for soft_clip_type in self.soft_clip_types:
                if soft_clip_type == "sox":
                    self.soft_clip_perturbers.append(
                        SpeakerDistortionPerturbSox(sample_rate=self.sampling_rate)
                    )
                elif soft_clip_type == "pedal":
                    self.soft_clip_perturbers.append(
                        SpeakerDistortionPerturbPedal(sample_rate=self.sampling_rate)
                    )
                elif soft_clip_type == "soft":
                    self.soft_clip_perturbers.append(
                        SpeakerDistortionPerturbSoftClip(sample_rate=self.sampling_rate)
                    )
                elif soft_clip_type == "sigmoid1":
                    self.soft_clip_perturbers.append(
                        SpeakerDistortionPerturbSigmoid1(sample_rate=self.sampling_rate)
                    )
                elif soft_clip_type == "sigmoid2":
                    self.soft_clip_perturbers.append(
                        SpeakerDistortionPerturbSigmoid2(sample_rate=self.sampling_rate)
                    )

        # EQ perturb
        self.eq_perturb_prob = eq_perturb_prob
        self.use_eq_perturb = bool(self.eq_perturb_prob > 0)
        if self.use_eq_perturb:
            self.eq_perturber = EQPerturb(
                sample_rate=self.sampling_rate, db_min=eq_db_min, db_max=eq_db_max
            )

        # EQ much gain perturb
        self.eq_much_gain_prob = eq_much_gain_prob
        if self.eq_much_gain_prob > 0:
            self.eq_much_gain_perturber = EQMuchGainPerturb(
                sample_rate=self.sampling_rate,
                db_min=eq_much_gain_db_min,
                db_max=eq_much_gain_db_max,
                freq_min=eq_much_gain_freq_min,
                freq_max=eq_much_gain_freq_max,
            )

        # band reject perturb
        self.band_reject_prob = band_reject_prob
        if self.band_reject_prob > 0:
            self.band_reject_perturber = BandRejectPerturb(
                sample_rate=self.sampling_rate,
                min_center_freq=band_reject_min_center_freq,
                max_center_freq=band_reject_max_center_freq,
                min_q=band_reject_min_q,
                max_q=band_reject_max_q,
                min_freq_bandwidth=band_reject_min_freq_bandwidth,
                max_freq_bandwidth=band_reject_max_freq_bandwidth,
                use_stft=band_reject_use_stft,
                max_n=band_reject_max_n,
            )

        # bass boost perturb
        self.bass_boost_prob = bass_boost_prob
        if self.bass_boost_prob > 0:
            self.bass_boost_perturber = BassBoostPerturb(
                sample_rate=self.sampling_rate,
                highpass_cutoff_min=bass_boost_highpass_cutoff_min,
                highpass_cutoff_max=bass_boost_highpass_cutoff_max,
                attenuation_min_db=bass_boost_attenuation_min_db,
            )

        # DC offset perturb
        self.dc_offset_prob = dc_offset_prob
        if self.dc_offset_prob > 0:
            self.dc_offset_perturber = DCOffsetPerturb(
                sample_rate=self.sampling_rate, min_offset=dc_offset_min, max_offset=dc_offset_max
            )

        # spectral leakage perturb
        self.spectral_leakage_prob = spectral_leakage_prob
        if self.spectral_leakage_prob > 0:
            self.spectral_leakage_perturber = SpectralLeakagePerturb(
                sample_rate=self.sampling_rate,
                window_lengths=spectral_leakage_window_lengths,
                max_time_shift=spectral_leakage_max_time_shift,
            )

        # colored noise perturb
        self.colored_noise_prob = colored_noise_prob
        if self.colored_noise_prob > 0:
            self.colored_noise_perturber = ColoredNoisePerturb(
                sample_rate=self.sampling_rate,
                snr_min=colered_noise_snr_min,
                snr_max=colered_noise_snr_max,
                color_types=colered_noise_types,
            )

        # low pass perturb
        self.lowpass_prob = lowpass_prob
        if self.lowpass_prob > 0:
            self.lowpass_perturber = LowPassPerturb(
                sample_rate=self.sampling_rate,
                min_cutoff_freq=lowpass_min_cutoff_freq,
                max_cutoff_freq=lowpass_max_cutoff_freq,
                min_order=lowpass_min_order,
                max_order=lowpass_max_order,
            )

        # spectral Time Frequency Holes perturb
        self.spectral_time_freq_holes_prob = spectral_time_freq_holes_prob
        if self.spectral_time_freq_holes_prob > 0:
            self.spectral_time_freq_holes_perturber = SpectralTimeFreqHolesPerturb(
                sample_rate=self.sampling_rate,
                stft_frame_length=spectral_time_freq_holes_stft_frame_length,
                stft_frame_step=spectral_time_freq_holes_stft_frame_step,
                holes_num_min=spectral_time_freq_holes_stft_holes_num_min,
                holes_num_max=spectral_time_freq_holes_stft_holes_num_max,
                holes_width_min_freq=spectral_time_freq_holes_stft_holes_width_min_freq,
                holes_width_max_freq=spectral_time_freq_holes_stft_holes_width_max_freq,
                holes_width_min_time=spectral_time_freq_holes_stft_holes_width_min_time,
                holes_width_max_time=spectral_time_freq_holes_stft_holes_width_max_time,
                cutoff_freq=spectral_time_freq_holes_cutoff_freq,
            )

        # webrtc ns perturb
        self.webrtc_ns_prob = webrtc_ns_prob
        if self.webrtc_ns_prob > 0:
            self.webrtc_ns_perturber = WebRTCNS_perturb(
                sample_rate=self.sampling_rate, channels=1, ns_levels=webrtc_ns_levels
            )
        self.webrtc_ns_volume_protection = webrtc_ns_volume_protection

        # webrtc agc perturb
        self.webrtc_agc_prob = webrtc_agc_prob
        if self.webrtc_agc_prob > 0:
            target_level_dbfs_list = list(
                range(webrtc_agc_target_level_dbfs_min, webrtc_agc_target_level_dbfs_max + 1)
            )
            self.webrtc_agc_perturber = WebRTCSAGC_perturb(
                sample_rate=self.sampling_rate,
                channels=1,
                target_level_dbfs_list=target_level_dbfs_list,
            )

        # drc perturb
        self.drc_prob = drc_prob
        if self.drc_prob > 0:
            self.drc_perturber = DRCPerturb(
                sample_rate=self.sampling_rate,
                threshold_db_min=drc_threshold_db_min,
                threshold_db_max=drc_threshold_db_max,
                ratio_min=drc_ratio_min,
                ratio_max=drc_ratio_max,
                attack_ms_min=drc_attack_ms_min,
                attack_ms_max=drc_attack_ms_max,
                release_ms_min=drc_release_ms_min,
                release_ms_max=drc_release_ms_max,
            )

        # codecs perturb
        self.codecs_prob = codecs_prob
        if self.codecs_prob > 0:
            self.codecs_types = codecs_types
            self.codecs_perturbers = list()
            self.codecs_perturbers_prob = list()
            for codecs_type in self.codecs_types:
                if codecs_type == "mp3":
                    self.codecs_perturbers.append(
                        MP3CompressorPerturb(sample_rate=self.sampling_rate)
                    )
                    self.codecs_perturbers_prob.append(0.4)
                elif codecs_type == "aac":
                    self.codecs_perturbers.append(
                        AACConversionPerturb(sample_rate=self.sampling_rate)
                    )
                    self.codecs_perturbers_prob.append(0.1)
                elif codecs_type == "gsm":
                    self.codecs_perturbers.append(GSMcodecsPerturb(sample_rate=self.sampling_rate))
                    self.codecs_perturbers_prob.append(0.1)
                elif codecs_type == "opus":
                    self.codecs_perturbers.append(
                        OPUSCodecsPerturb(sample_rate=self.sampling_rate)
                    )
                    self.codecs_perturbers_prob.append(0.4)

            self.codecs_perturbers_prob = np.array(self.codecs_perturbers_prob) / np.sum(
                self.codecs_perturbers_prob
            )

        # packet loss perturb
        self.packet_loss_prob = packet_loss_prob
        if self.packet_loss_prob > 0:
            self.packet_loss_perturber = PacketLossPerturb(
                sample_rate=self.sampling_rate,
                loss_rate_min=packet_loss_rate_min,
                loss_rate_max=packet_loss_rate_max,
                frame_time_min=packet_loss_frame_time_min,
                frame_time_max=packet_loss_frame_time_max,
                decay_rate_min=packet_loss_decay_rate_min,
                decay_rate_max=packet_loss_decay_rate_max,
                hard_loss_prob=packet_loss_hard_loss_prob,
                loss_on_vad=packet_loss_on_vad,
            )

        # bit crush perturb
        self.bit_crush_prob = bit_crush_prob
        if self.bit_crush_prob > 0:
            self.bit_crush_perturber = BitCrushPerturb(
                sample_rate=self.sampling_rate,
                bit_min=bit_crush_bit_min,
                bit_max=bit_crush_bit_max,
            )

        # colored noise post perturb
        self.colored_noise_post_prob = colored_noise_post_prob
        if self.colored_noise_post_prob > 0:
            self.colored_noise_post_perturber = ColoredNoisePerturb(
                sample_rate=self.sampling_rate,
                snr_min=colored_noise_post_snr_min,
                snr_max=colored_noise_post_snr_max,
                color_types=colored_noise_post_types,
            )

        # output volume range
        assert random_volume is not None
        if random_volume:
            assert (
                volume_min_dB is not None
                and volume_max_dB is not None
                or volume_min_sample is not None
                and volume_max_sample is not None
            )
        self.use_random_volume = random_volume
        self.volume_min_dB = volume_min_dB
        self.volume_max_dB = volume_max_dB
        self.volume_min_sample = volume_min_sample
        self.volume_max_sample = volume_max_sample
        self.use_rms_volume = use_rms_volume
        self.sync_random_volume = sync_random_volume

        # output cut
        self.output_cut_seconds = output_cut_seconds
        self.output_random_cut = output_random_cut
        self.output_normalize = output_normalize
        self.output_resample = output_resample
        self.output_resample_rate = output_resample_rate

        # debug
        self.debug = debug
        self.dummy = dummy

        if self.rank == 0:
            print("dataset initialized")
            if hasattr(self, "clean_list"):
                print(
                    f"length of clean speech list: {len(self.clean_list)}, length of noise list: {len(self.noise_list)}"
                )

    def __len__(self):
        if self.dummy:
            return 100
        if hasattr(self, "length"):
            return self.length
        if hasattr(self, "collection"):
            return len(self.collection)
        if hasattr(self, "clean_list"):
            return len(self.clean_list)
        return None

    # @profile
    def __getitem__(self, idx):
        output_dict = OrderedDict()
        # get clean data
        clean_data_output_dict = self.get_clean(idx)
        for key in clean_data_output_dict:
            output_dict[key] = clean_data_output_dict[key]
        clean_data = clean_data_output_dict["perturbed_clean"]
        clean_data = np.nan_to_num(clean_data, nan=0, posinf=0, neginf=0)
        if self.debug:
            output_dict["original_clean"] = clean_data.astype(np.float32)
        # get noise data
        add_noise_flag = np.random.random() < self.add_noise_prob
        only_noise_flag = np.random.random() < self.only_noise_prob
        if add_noise_flag or only_noise_flag:
            if self.trim_noise:
                noise_data = self.get_noise(length=clean_data.shape[0])
            else:
                noise_data = self.get_noise(length=None)
        else:
            noise_data = np.zeros_like(clean_data)
        noise_data = np.nan_to_num(noise_data, nan=0, posinf=0, neginf=0)
        if self.debug:
            output_dict["original_noise"] = noise_data.astype(np.float32)
        # reverbrate
        clean_reverb_flag = np.random.random() < self.reverb_prob
        # noise_reverb_flag = self.reverb_noise and random.random() < self.reverb_prob
        if clean_reverb_flag:
            clean_data_reverb, clean_data_reverb_early = self.reverberate(clean_data)
            clean_data = clean_data_reverb_early
            if self.debug:
                output_dict["reverb_clean"] = clean_data_reverb.astype(np.float32)
                output_dict["reverb_clean_early"] = clean_data_reverb_early.astype(np.float32)
        else:
            clean_data_reverb = clean_data.copy()
        # if noise_reverb_flag:
        #     noise_data = self.reverberate(noise_data)
        #     output_dict["reverb_noise"] = noise_data.astype(np.float32)
        # add noise
        if only_noise_flag:
            noisy_data = noise_data.copy()
            clean_data_reverb = np.zeros_like(noise_data)
            snr = -1000
        elif add_noise_flag:
            noisy_data, clean_data_reverb, noise_data, snr = self.add_noise(
                clean_data_reverb, noise_data
            )
        else:
            noisy_data = clean_data_reverb.copy()
            snr = np.inf
        if self.debug:
            output_dict["addnoise_clean"] = clean_data_reverb.astype(np.float32)
            output_dict["addnoise_noise"] = noise_data.astype(np.float32)
            output_dict["addnoise_noisy"] = noisy_data.astype(np.float32)
        output_dict["SNR"] = snr
        # loudness perturb
        perturbed_data = noisy_data
        loudness_perturb_flag = np.random.random() < self.loudness_perturb_prob
        if loudness_perturb_flag:
            perturbed_data = self.loudness_perturber(perturbed_data)
            if self.debug:
                output_dict["loudness_perturbed"] = perturbed_data.astype(np.float32)
        # clipping perturb
        clip_perturb_flag = np.random.random() < self.clip_prob
        if clip_perturb_flag:
            hard_clip_flag = np.random.random() < self.hard_clip_portion
            if hard_clip_flag:
                perturbed_data = self.hard_clip_perturber(perturbed_data)
            else:
                perturber = np.random.choice(self.soft_clip_perturbers)
                perturbed_data = perturber(perturbed_data)
            if self.debug:
                output_dict["clip_perturbed"] = perturbed_data.astype(np.float32)
        # EQ perturb
        eq_perturb_flag = np.random.random() < self.eq_perturb_prob
        if eq_perturb_flag:
            perturbed_data = self.eq_perturber(perturbed_data)
            if self.debug:
                output_dict["eq_perturbed"] = perturbed_data.astype(np.float32)
        # EQ much gain perturb
        eq_much_gain_perturb_flag = (
            np.random.random() < self.eq_much_gain_prob and not eq_perturb_flag
        )
        if eq_much_gain_perturb_flag:
            perturbed_data = self.eq_much_gain_perturber(perturbed_data)
            if self.debug:
                output_dict["eq_much_gain_perturbed"] = perturbed_data.astype(np.float32)
        # band reject perturb
        band_reject_perturb_flag = np.random.random() < self.band_reject_prob
        if band_reject_perturb_flag:
            perturbed_data = self.band_reject_perturber(perturbed_data)
            if self.debug:
                output_dict["band_reject_perturbed"] = perturbed_data.astype(np.float32)
        # bass boost perturb
        bass_boost_perturb_flag = (
            np.random.random() < self.bass_boost_prob
            and not eq_perturb_flag
            and not eq_much_gain_perturb_flag
        )
        if bass_boost_perturb_flag:
            perturbed_data = self.bass_boost_perturber(perturbed_data)
            if self.debug:
                output_dict["bass_boost_perturbed"] = perturbed_data.astype(np.float32)
        # DC offset perturb
        dc_offset_perturb_flag = np.random.random() < self.dc_offset_prob
        if dc_offset_perturb_flag:
            perturbed_data = self.dc_offset_perturber(perturbed_data)
            if self.debug:
                output_dict["dc_offset_perturbed"] = perturbed_data.astype(np.float32)
        # spectral leakage perturb
        spectral_leakage_perturb_flag = np.random.random() < self.spectral_leakage_prob
        if spectral_leakage_perturb_flag:
            perturbed_data = self.spectral_leakage_perturber(perturbed_data)
            if self.debug:
                output_dict["spectral_leakage_perturbed"] = perturbed_data.astype(np.float32)
        # colored noise perturb
        colored_noise_perturb_flag = np.random.random() < self.colored_noise_prob
        if colored_noise_perturb_flag:
            perturbed_data = self.colored_noise_perturber(perturbed_data)
            if self.debug:
                output_dict["colored_noise_perturbed"] = perturbed_data.astype(np.float32)
        # low pass perturb
        lowpass_perturb_flag = np.random.random() < self.lowpass_prob
        if lowpass_perturb_flag:
            perturbed_data = self.lowpass_perturber(perturbed_data)
            if self.debug:
                output_dict["lowpass_perturbed"] = perturbed_data.astype(np.float32)
        # speatra time freq perturb
        spectral_time_freq_holes_perturb_flag = (
            np.random.random() < self.spectral_time_freq_holes_prob
        )
        if spectral_time_freq_holes_perturb_flag:
            perturbed_data = self.spectral_time_freq_holes_perturber(perturbed_data)
            if self.debug:
                output_dict["spectral_time_freq_holes_perturbed"] = perturbed_data.astype(
                    np.float32
                )
        # webrtc ns perturb
        webrtc_ns_perturb_flag = np.random.random() < self.webrtc_ns_prob
        if webrtc_ns_perturb_flag:
            if self.webrtc_ns_volume_protection:
                if np.abs(perturbed_data).max() > 0.99:
                    perturbed_data = perturbed_data / np.abs(perturbed_data).max() * 0.99
                    clean_data = clean_data / np.abs(clean_data).max() * 0.99
            perturbed_data = self.webrtc_ns_perturber(perturbed_data)
            # if self.sampling_rate == 48000:
            #     perturbed_data = perturbed_data[334:]
            #     clean_data = clean_data[:-334]
            if self.debug:
                output_dict["webrtc_ns_perturbed"] = perturbed_data.astype(np.float32)
        # webrtc agc perturb
        webrtc_agc_perturb_flag = np.random.random() < self.webrtc_agc_prob
        if webrtc_agc_perturb_flag:
            perturbed_data = self.webrtc_agc_perturber(perturbed_data)
            if self.debug:
                output_dict["webrtc_agc_perturbed"] = perturbed_data.astype(np.float32)
        # drc perturb
        drc_perturb_flag = np.random.random() < self.drc_prob
        if drc_perturb_flag:
            perturbed_data = self.drc_perturber(perturbed_data)
            if self.debug:
                output_dict["drc_perturbed"] = perturbed_data.astype(np.float32)
        # codecs perturb
        codecs_perturb_flag = np.random.random() < self.codecs_prob
        if codecs_perturb_flag:
            codecs_perturber = np.random.choice(
                self.codecs_perturbers, p=self.codecs_perturbers_prob
            )
            perturbed_data = codecs_perturber(perturbed_data)
            codecs_name = codecs_perturber.name
            # if codecs_name == "AAC":
            # perturbed_data = perturbed_data[len(perturbed_data)-len(clean_data):]
            if self.debug:
                output_dict[f"codecs_perturbed_{codecs_name}"] = perturbed_data.astype(np.float32)
        # packet loss perturb
        packet_loss_perturb_flag = np.random.random() < self.packet_loss_prob
        if packet_loss_perturb_flag:
            perturbed_data = self.packet_loss_perturber(perturbed_data)
            if self.debug:
                output_dict["packet_loss_perturbed"] = perturbed_data.astype(np.float32)
        # bit crush perturb
        bit_crush_perturb_flag = np.random.random() < self.bit_crush_prob
        if bit_crush_perturb_flag:
            perturbed_data = self.bit_crush_perturber(perturbed_data)
            if self.debug:
                output_dict["bit_crush_perturbed"] = perturbed_data.astype(np.float32)
        # colored noise post perturb
        colored_noise_post_perturb_flag = np.random.random() < self.colored_noise_post_prob
        if colored_noise_post_perturb_flag:
            perturbed_data = self.colored_noise_post_perturber(perturbed_data)
            if self.debug:
                output_dict["colored_noise_post_perturbed"] = perturbed_data.astype(np.float32)
        # volume augmentation
        if self.use_random_volume:
            if self.sync_random_volume:
                perturbed_data, clean_data, target_volume_perturbed = self.random_volume_dual(
                    perturbed_data, clean_data
                )
                target_volume_clean = target_volume_perturbed
                perturbed_data, clean_data = self.volume_clip_dual(perturbed_data, clean_data)
            else:
                perturbed_data, target_volume_perturbed = self.random_volume(perturbed_data)
                perturbed_data = self.volume_clip(perturbed_data)
                clean_data, target_volume_clean = self.random_volume(clean_data)
                clean_data = self.volume_clip(clean_data)
            if self.debug:
                output_dict["random_volume_clean"] = clean_data.astype(np.float32)
                output_dict["random_volume_perturbed"] = perturbed_data.astype(np.float32)

        output_dict["perturbed"] = perturbed_data.astype(np.float32)

        output_dict["clean"] = clean_data.astype(np.float32)

        output_dict["name"] = f"index{idx}"

        output_dict["sampling_rate"] = self.sampling_rate

        if self.use_random_volume:
            output_dict["target_volume_perturbed"] = target_volume_perturbed
            output_dict["target_volume_clean"] = target_volume_clean
        output_dict["n_speakers"] = self.n_speakers
        if len(output_dict["perturbed"]) > len(output_dict["clean"]):
            output_dict["perturbed"] = output_dict["perturbed"][: len(output_dict["clean"])]
        if len(output_dict["clean"]) > len(output_dict["perturbed"]):
            output_dict["clean"] = output_dict["clean"][: len(output_dict["perturbed"])]

        if self.output_cut_seconds:
            if self.output_random_cut:
                start = np.random.randint(
                    0,
                    len(output_dict["perturbed"])
                    - self.output_cut_seconds * self.sampling_rate
                    + 1,
                )
            else:
                start = 0
            end = start + self.output_cut_seconds * self.sampling_rate
            output_dict["perturbed"] = output_dict["perturbed"][start:end]
            output_dict["clean"] = output_dict["clean"][start:end]
            if len(output_dict["perturbed"]) < self.output_cut_seconds * self.sampling_rate:
                output_dict["perturbed"] = np.pad(
                    output_dict["perturbed"],
                    (
                        0,
                        self.output_cut_seconds * self.sampling_rate
                        - len(output_dict["perturbed"]),
                    ),
                    mode="constant",
                )
                output_dict["clean"] = np.pad(
                    output_dict["clean"],
                    (0, self.output_cut_seconds * self.sampling_rate - len(output_dict["clean"])),
                    mode="constant",
                )

        if self.output_normalize:
            norm_factor = max(
                np.max(np.abs(output_dict["perturbed"]), axis=0),
                np.max(np.abs(output_dict["clean"]), axis=0),
            )
            output_dict["perturbed"] = output_dict["perturbed"] / norm_factor * 0.8
            output_dict["clean"] = output_dict["clean"] / norm_factor * 0.8

        if self.output_resample:
            output_dict["perturbed"] = self.resample_signal(
                output_dict["perturbed"],
                self.sampling_rate,
                self.output_resample_rate,
                self.resample_method,
            )
            # output_dict["clean"] = self.resample_signal(output_dict["clean"], self.sampling_rate, self.output_resample_rate, self.resample_method)
            output_dict["sampling_rate"] = self.output_resample_rate

        gc.collect()
        return output_dict

    @staticmethod
    def pad_head_tail(y, n_fft, pad_mode="reflect"):
        pad = int(n_fft // 2)
        y = np.pad(y, (pad, pad), mode=pad_mode)
        return y

    @staticmethod
    def vad_merge(w):
        intervals = librosa.effects.split(w, top_db=50)
        temp = list()
        for s, e in intervals:
            temp.append(w[s:e])
        if len(temp) == 0:
            return w
        return np.concatenate(temp, axis=None)

    @staticmethod
    def resample_signal(input_signal, orig_sample_rate, dest_sample_rate, resample_method):
        return librosa.resample(
            input_signal,
            orig_sr=orig_sample_rate,
            target_sr=dest_sample_rate,
            res_type=resample_method,
        )

    def parse_json_double(self, clean_json_path, noise_json_path):
        clean_list = []
        clean_duration_list = []
        noise_list = []
        with open(clean_json_path) as f:
            for one_line in f.readlines():
                json_data = json.loads(one_line.strip())
                duration = json_data["duration"]
                sample_rate = json_data["sample_rate"]
                if self.min_duration_seconds and self.max_duration_seconds:
                    if self.min_duration_seconds < duration < self.max_duration_seconds:
                        clean_list.append(json_data["file_path"])
                        clean_duration_list.append(float(duration))
                elif self.min_duration_seconds:
                    if self.min_duration_seconds < duration:
                        clean_list.append(json_data["file_path"])
                        clean_duration_list.append(float(duration))
                elif self.max_duration_seconds:
                    if duration < self.max_duration_seconds:
                        clean_list.append(json_data["file_path"])
                        clean_duration_list.append(float(duration))
                else:
                    clean_list.append(json_data["file_path"])
                    clean_duration_list.append(float(duration))
        with open(noise_json_path) as f:
            for one_line in f.readlines():
                json_data = json.loads(one_line.strip())
                sample_rate = json_data["sample_rate"]
                noise_list.append(json_data["file_path"])
        if self.rank == 0:
            print(
                f"clean speech utterance number: {len(clean_duration_list)}, total duration: {sum(clean_duration_list)/3600.} hours"
            )
        return clean_list, clean_duration_list, noise_list

    def parse_list(self, clean_list_path, noise_list_path):
        orig_clean_list = [line.rstrip("\n") for line in open(clean_list_path)]
        orig_noise_list = [line.rstrip("\n") for line in open(noise_list_path)]
        clean_list = []
        clean_duration_list = []
        noise_list = []
        noise_duration_list = []
        if self.min_duration_seconds and self.max_duration_seconds:
            for clean_path in orig_clean_list:
                duration = librosa.get_duration(filename=clean_path, sr=None)
                if self.min_duration_seconds < duration < self.max_duration_seconds:
                    if self.check_list_files:
                        try:
                            sf.read(clean_path)
                        except Exception as e:
                            print(f"error: {e}, noise path: {clean_path}")
                            continue
                    clean_list.append(clean_path)
                    clean_duration_list.append(duration)
        elif self.min_duration_seconds:
            for clean_path in orig_clean_list:
                duration = librosa.get_duration(filename=clean_path, sr=None)
                if duration > self.min_duration_seconds:
                    if self.check_list_files:
                        try:
                            sf.read(clean_path)
                        except Exception as e:
                            print(f"error: {e}, noise path: {clean_path}")
                            continue
                    clean_list.append(clean_path)
                    clean_duration_list.append(duration)
        elif self.max_duration_seconds:
            for clean_path in orig_clean_list:
                duration = librosa.get_duration(filename=clean_path, sr=None)
                if duration < self.max_duration_seconds:
                    if self.check_list_files:
                        try:
                            sf.read(clean_path)
                        except Exception as e:
                            print(f"error: {e}, noise path: {clean_path}")
                            continue
                    clean_list.append(clean_path)
                    clean_duration_list.append(duration)
        else:
            for clean_path in orig_clean_list:
                duration = librosa.get_duration(filename=clean_path, sr=None)
                if self.check_list_files:
                    try:
                        sf.read(clean_path)
                    except Exception as e:
                        print(f"error: {e}, noise path: {clean_path}")
                        continue
                clean_list.append(clean_path)
                clean_duration_list.append(duration)
        for noise_path in orig_noise_list:
            duration = librosa.get_duration(filename=clean_path, sr=None)
            if self.check_list_files:
                try:
                    sf.read(noise_path)
                except Exception as e:
                    print(f"error: {e}, noise path: {noise_path}")
                    continue
            noise_list.append(noise_path)
            noise_duration_list.append(duration)
        if self.rank == 0:
            print(
                f"clean speech utterance number: {len(clean_duration_list)}, total duration: {sum(clean_duration_list)/3600.} hours"
            )
            print(
                f"noise utterance number: {len(noise_duration_list)}, total duration: {sum(noise_duration_list)/3600.} hours"
            )
        return clean_list, clean_duration_list, noise_list

    def get_clean(self, idx):
        # read index speech
        speed_perturb_flag = np.random.random() < self.speed_perturb_prob
        pitch_shift_flag = np.random.random() < self.pitch_shift_prob
        add_extra_space_flag = np.random.random() < self.add_extra_space_prob
        self.n_speakers = n_speakers = np.random.randint(
            self.min_n_speakers, self.max_n_speakers + 1
        )
        file_path = self.clean_list[idx]
        temp, sr = sf.read(file_path)
        if temp.ndim > 1:
            temp = temp[:, 0]
        if self.remove_dc_offset:
            temp = temp - temp.mean()
        # assert(sr >= self.sampling_rate)
        try:
            librosa.util.valid_audio(temp, mono=True)
        except Exception as e:
            print(file_path, e)
            temp = np.zeros_like(temp)
        if sr != self.sampling_rate:
            temp = self.resample_signal(temp, sr, self.sampling_rate, self.resample_method)
            try:
                librosa.util.valid_audio(temp, mono=True)
            except Exception as e:
                print(f"resample error: {file_path}, {e}")
                temp = np.zeros_like(temp)
        clean_data = temp.astype(np.float32)
        if n_speakers > 1:
            cur_rest_speakers = n_speakers - 1
            target_volume_for_add = np.sqrt(np.mean(self.vad_merge(clean_data) ** 2) + 1e-8)
            while cur_rest_speakers > 0:
                file_path = np.random.choice(self.clean_list)
                temp, sr = sf.read(file_path)
                if temp.ndim > 1:
                    temp = temp[:, 0]
                if self.remove_dc_offset:
                    temp = temp - temp.mean()
                # assert(sr >= self.sampling_rate)
                try:
                    librosa.util.valid_audio(temp, mono=True)
                except Exception as e:
                    print(file_path, e)
                    temp = np.zeros_like(temp)
                if sr != self.sampling_rate:
                    temp = self.resample_signal(temp, sr, self.sampling_rate, self.resample_method)
                    try:
                        librosa.util.valid_audio(temp, mono=True)
                    except Exception as e:
                        print("resample error", file_path, e)
                        temp = np.zeros_like(temp)
                clean_data_add = temp.astype(np.float32)
                volume = np.sqrt(np.mean(self.vad_merge(clean_data_add) ** 2) + 1e-8)
                volume_perturb = np.random.uniform(
                    self.speech_splice_equal_volume_range[0],
                    self.speech_splice_equal_volume_range[1],
                )
                target_volume_tmp = target_volume_for_add * 10 ** (volume_perturb / 20)
                clean_data_add = clean_data_add * target_volume_tmp / volume
                if len(clean_data) < len(clean_data_add):
                    clean_data = np.pad(
                        clean_data, (0, len(clean_data_add) - len(clean_data)), "constant"
                    )
                elif len(clean_data_add) < len(clean_data):
                    clean_data_add = np.pad(
                        clean_data_add, (0, len(clean_data) - len(clean_data_add)), "constant"
                    )
                clean_data = clean_data + clean_data_add
                cur_rest_speakers -= 1
        # clean_data = clean_data - clean_data.mean()
        # if self.debug:
        #     clean_data_before_speed = clean_data.copy()
        # if speed_perturb_flag:
        #     # print("speed perturb")
        #     clean_data, _ = self.speed_perturber.process(clean_data)
        # # if self.debug:
        # #     clean_data_before_pitch = clean_data.copy()
        # if pitch_shift_flag:
        #     # print("pitch shift")
        #     clean_data, _ = self.pitch_shifter.process(clean_data)
        # if self.debug:
        #     clean_data_before_eq = clean_data.copy()
        if add_extra_space_flag:
            head_paddding_size = np.random.randint(0, int(0.3 * self.sampling_rate) + 1) * int(
                np.random.random() < 0.8
            )
            tail_padding_size = np.random.randint(0, int(0.3 * self.sampling_rate) + 1) * int(
                np.random.random() < 0.8
            )
            # head_paddding_size = int(1*self.sampling_rate)
            # tail_padding_size = int(1*self.sampling_rate)
            clean_data = np.pad(clean_data, (head_paddding_size, tail_padding_size), "constant")
        # read random speech and splice
        speech_splice_flag = self.speech_splice
        if speech_splice_flag:
            if self.speech_splice_equal_volume:
                target_volume = np.sqrt(np.mean(self.vad_merge(clean_data) ** 2) + 1e-8)
            val_len = self.speech_splice_length
            while len(clean_data) < val_len:
                speed_perturb_flag = np.random.random() < self.speed_perturb_prob
                pitch_shift_flag = np.random.random() < self.pitch_shift_prob
                add_extra_space_flag = np.random.random() < self.add_extra_space_prob
                file_path = np.random.choice(self.clean_list)
                temp, sr = sf.read(file_path)
                if temp.ndim > 1:
                    temp = temp[:, 0]
                if self.remove_dc_offset:
                    temp = temp - temp.mean()
                # assert(sr >= self.sampling_rate)
                try:
                    librosa.util.valid_audio(temp, mono=True)
                except Exception as e:
                    print(file_path, e)
                    temp = np.zeros_like(temp)
                if sr != self.sampling_rate:
                    temp = self.resample_signal(temp, sr, self.sampling_rate, self.resample_method)
                    try:
                        librosa.util.valid_audio(temp, mono=True)
                    except Exception as e:
                        print("resample error", file_path, e)
                        temp = np.zeros_like(temp)
                clean_data_cat = temp.astype(np.float32)
                if n_speakers > 1:
                    cur_rest_speakers = n_speakers - 1
                    target_volume_for_add = np.sqrt(
                        np.mean(self.vad_merge(clean_data_cat) ** 2) + 1e-8
                    )
                    while cur_rest_speakers > 0:
                        file_path = np.random.choice(self.clean_list)
                        temp, sr = sf.read(file_path)
                        if temp.ndim > 1:
                            temp = temp[:, 0]
                        if self.remove_dc_offset:
                            temp = temp - temp.mean()
                        # assert(sr >= self.sampling_rate)
                        try:
                            librosa.util.valid_audio(temp, mono=True)
                        except Exception as e:
                            print(file_path, e)
                            temp = np.zeros_like(temp)
                        if sr != self.sampling_rate:
                            temp = self.resample_signal(
                                temp, sr, self.sampling_rate, self.resample_method
                            )
                            try:
                                librosa.util.valid_audio(temp, mono=True)
                            except Exception as e:
                                print("resample error", file_path, e)
                                temp = np.zeros_like(temp)
                        clean_data_add = temp.astype(np.float32)
                        volume = np.sqrt(np.mean(self.vad_merge(clean_data_add) ** 2) + 1e-8)
                        volume_perturb = np.random.uniform(
                            self.speech_splice_equal_volume_range[0],
                            self.speech_splice_equal_volume_range[1],
                        )
                        target_volume_tmp = target_volume_for_add * 10 ** (volume_perturb / 20)
                        clean_data_add = clean_data_add * target_volume_tmp / volume
                        if len(clean_data_cat) < len(clean_data_add):
                            clean_data_cat = np.pad(
                                clean_data_cat,
                                (0, len(clean_data_add) - len(clean_data_cat)),
                                "constant",
                            )
                        elif len(clean_data_add) < len(clean_data_cat):
                            clean_data_add = np.pad(
                                clean_data_add,
                                (0, len(clean_data_cat) - len(clean_data_add)),
                                "constant",
                            )
                        clean_data_cat = clean_data_cat + clean_data_add
                        cur_rest_speakers -= 1
                # clean_data_cat = clean_data_cat - clean_data_cat.mean()
                # if self.debug:
                #     clean_data_cat_before_speed = clean_data_cat.copy()
                # if speed_perturb_flag:
                #     clean_data_cat, _ = self.speed_perturber.process(clean_data_cat)
                # # if self.debug:
                # #     clean_data_cat_before_pitch = clean_data_cat.copy()
                # if pitch_shift_flag:
                #     clean_data_cat, _ = self.pitch_shifter.process(clean_data_cat)
                # if self.debug:
                #     clean_data_cat_before_eq = clean_data_cat.copy()
                if add_extra_space_flag:
                    head_paddding_size = np.random.randint(
                        0, int(0.3 * self.sampling_rate) + 1
                    ) * int(np.random.random() < 0.8)
                    tail_padding_size = np.random.randint(
                        0, int(0.3 * self.sampling_rate) + 1
                    ) * int(np.random.random() < 0.8)
                    # head_paddding_size = int(1*self.sampling_rate)
                    # tail_padding_size = int(1*self.sampling_rate)
                    clean_data_cat = np.pad(
                        clean_data_cat, (head_paddding_size, tail_padding_size), "constant"
                    )
                if self.speech_splice_equal_volume:
                    volume = np.sqrt(np.mean(self.vad_merge(clean_data_cat) ** 2) + 1e-8)
                    volume_perturb = np.random.uniform(
                        self.speech_splice_equal_volume_range[0],
                        self.speech_splice_equal_volume_range[1],
                    )
                    target_volume_tmp = target_volume * 10 ** (volume_perturb / 20)
                    clean_data_cat = clean_data_cat * target_volume_tmp / volume
                clean_data = np.concatenate((clean_data, clean_data_cat))
                # if self.debug:
                #     clean_data_before_speed = np.concatenate((clean_data_before_speed, clean_data_cat_before_speed))
                #     clean_data_before_pitch = np.concatenate((clean_data_before_pitch, clean_data_cat_before_pitch))
            if len(clean_data) > val_len:
                if self.speech_random_start:
                    clean_start = np.random.randint(0, len(clean_data) - val_len + 1)
                else:
                    clean_start = 0
                clean_data = clean_data[clean_start : clean_start + val_len]
        output_dict = OrderedDict()
        output_dict["no_perturbed_clean"] = clean_data.astype(np.float32)
        if speed_perturb_flag:
            clean_data, _ = self.speed_perturber.process(clean_data)
        # if self.debug:
        #     clean_data_cat_before_pitch = clean_data_cat.copy()
        if pitch_shift_flag:
            clean_data, _ = self.pitch_shifter.process(clean_data)
        output_dict["perturbed_clean"] = clean_data.astype(np.float32)
        # if self.debug and self.clean_data_vad:
        #     return clean_data, clean_data_before_speed, clean_data_before_pitch, vad_label
        return output_dict

    def get_noise(self, length=None, idx=None):
        noise_mix_flag = np.random.random() < self.noise_mix_prob
        file_path = np.random.choice(self.noise_list)
        # file_path = self.noise_list[idx]
        temp, sr = sf.read(file_path)
        if temp.ndim > 1:
            temp = temp[:, 0]
        # assert(sr >= self.sampling_rate)
        try:
            librosa.util.valid_audio(temp, mono=True)
        except Exception as e:
            print(file_path, e)
            temp = np.zeros_like(temp)
        if sr != self.sampling_rate:
            temp = self.resample_signal(temp, sr, self.sampling_rate, self.resample_method)
            try:
                librosa.util.valid_audio(temp, mono=True)
            except Exception as e:
                print("resample error", file_path, e)
                temp = np.zeros_like(temp)
        noise_data = temp.astype(np.float32)
        if noise_mix_flag:
            file_path = np.random.choice(self.noise_list)
            temp, sr = sf.read(file_path)
            if temp.ndim > 1:
                temp = temp[:, 0]
            # assert(sr >= self.sampling_rate)
            try:
                librosa.util.valid_audio(temp, mono=True)
            except Exception as e:
                print(file_path, e)
                temp = np.zeros_like(temp)
            if sr != self.sampling_rate:
                temp = self.resample_signal(temp, sr, self.sampling_rate, self.resample_method)
                try:
                    librosa.util.valid_audio(temp, mono=True)
                except Exception as e:
                    print("resample error", file_path, e)
                    temp = np.zeros_like(temp)
            noise_data_2 = temp.astype(np.float32)
            if noise_data_2.shape[0] < noise_data.shape[0]:
                noise_data_2 = np.pad(
                    noise_data_2, (0, noise_data.shape[0] - noise_data_2.shape[0]), "constant"
                )
            mix_ratio = np.random.uniform(0.1, 1.0)
            noise_data = noise_data + mix_ratio * noise_data_2[: noise_data.shape[0]]

        if length:
            while len(noise_data) < length:
                if not self.noise_repeat_splice:
                    file_path = np.random.choice(self.noise_list)
                    temp, sr = sf.read(file_path)
                    if temp.ndim > 1:
                        temp = temp[:, 0]
                    # assert(sr >= self.sampling_rate)
                    try:
                        librosa.util.valid_audio(temp, mono=True)
                    except Exception as e:
                        print(file_path, e)
                        temp = np.zeros_like(temp)
                    if sr != self.sampling_rate:
                        temp = self.resample_signal(
                            temp, sr, self.sampling_rate, self.resample_method
                        )
                        try:
                            librosa.util.valid_audio(temp, mono=True)
                        except Exception as e:
                            print("resample error", file_path, e)
                            temp = np.zeros_like(temp)
                    noise_data_cat = temp.astype(np.float32)
                    if noise_mix_flag:
                        file_path = np.random.choice(self.noise_list)
                        temp, sr = sf.read(file_path)
                        if temp.ndim > 1:
                            temp = temp[:, 0]
                        # assert(sr >= self.sampling_rate)
                        try:
                            librosa.util.valid_audio(temp, mono=True)
                        except Exception as e:
                            print(file_path, e)
                            temp = np.zeros_like(temp)
                        if sr != self.sampling_rate:
                            temp = self.resample_signal(
                                temp, sr, self.sampling_rate, self.resample_method
                            )
                            try:
                                librosa.util.valid_audio(temp, mono=True)
                            except Exception as e:
                                print("resample error", file_path, e)
                                temp = np.zeros_like(temp)
                        noise_data_2 = temp.astype(np.float32)
                        if noise_data_2.shape[0] < noise_data_cat.shape[0]:
                            noise_data_2 = np.pad(
                                noise_data_2,
                                (0, noise_data_cat.shape[0] - noise_data_2.shape[0]),
                                "constant",
                            )
                        mix_ratio = np.random.uniform(0.1, 1.0)
                        noise_data_cat = (
                            noise_data_cat + mix_ratio * noise_data_2[: noise_data_cat.shape[0]]
                        )
                else:
                    noise_data_cat = noise_data.copy()
                noise_data = np.concatenate((noise_data, noise_data_cat))
            if len(noise_data) > length:
                noise_start = np.random.randint(0, len(noise_data) - length + 1)
                noise_data = noise_data[noise_start : noise_start + length]
        return noise_data

    def get_rir(self):
        if self.reverb_use_FRA:
            rir, direct_rir = FRA_RIR(nsource=1, direct_range=[-6, 50], max_T60=0.05)
            rir_data = rir[0].numpy().astype(np.float32)
        else:
            rir_path = np.random.choice(self.rir_list)
            if self.min_rt60 and self.max_rt60:
                rt60 = rir_path.split("rt")[1].split("_")[0]
                while float(rt60) < self.min_rt60 or float(rt60) > self.max_rt60:
                    rir_path = np.random.choice(self.rir_list)
                    rt60 = rir_path.split("rt")[1].split("_")[0]
            with open(rir_path, "rb") as f:
                rir_data = pickle.load(f)
            if "source_rir" in rir_data.keys():
                rir_data = rir_data["source_rir"]
            elif "rir" in rir_data.keys():
                rir_data = rir_data["rir"]
            rir_data = rir_data[:, 0]
            rir_data = rir_data[np.argmax(np.abs(rir_data)) :]
            rir_data = rir_data / np.abs(rir_data).max()
        assert rir_data.ndim == 1
        rir_early_data = rir_data[:6]
        return rir_data, rir_early_data

    def reverberate(self, clean_data):
        rir_data, rir_early_data = self.get_rir()
        cut_length = clean_data.shape[0]
        clean_data_reverb = fftconvolve(clean_data, rir_data, mode="full")[:cut_length]
        clean_data_reverb_early = fftconvolve(clean_data, rir_early_data, mode="full")[:cut_length]
        return clean_data_reverb, clean_data_reverb_early

    def add_noise(self, clean_data, noise_data):
        snr = np.random.uniform(self.snr_min, self.snr_max)
        clean_power = np.mean(self.vad_merge(clean_data) ** 2)
        noise_power = np.mean(self.vad_merge(noise_data) ** 2)
        noise_scale = np.sqrt(
            clean_power / (noise_power + 1e-8) / np.power(10.0, snr / 10.0) + 1e-8
        )
        noise_data *= noise_scale
        noisy_data = clean_data + noise_data
        librosa.util.valid_audio(noisy_data, mono=True)
        return noisy_data, clean_data, noise_data, snr

    def random_volume(self, noisy_data):
        if self.volume_min_dB and self.volume_max_dB:
            target_volume_dB = np.random.uniform(self.volume_min_dB, self.volume_max_dB)
            target_volume = np.power(10.0, target_volume_dB / 20.0)
        elif self.volume_min_sample and self.volume_max_sample:
            target_volume = np.random.uniform(self.volume_min_sample, self.volume_max_sample)

        if self.use_rms_volume:
            noisy_volume = np.sqrt(np.mean(self.vad_merge(noisy_data) ** 2) + 1e-8)
        else:
            noisy_volume = np.abs(noisy_data).max()

        scale = target_volume / (noisy_volume + 1e-6)
        noisy_data *= scale

        return noisy_data, target_volume

    @staticmethod
    def volume_clip(noisy_data):
        noisy_volume = np.abs(noisy_data).max()
        if noisy_volume > 0.99:
            noisy_data *= 0.99 / noisy_volume
        return noisy_data

    def random_volume_dual(self, noisy_data, clean_data):
        if self.volume_min_dB and self.volume_max_dB:
            target_volume_dB = np.random.uniform(self.volume_min_dB, self.volume_max_dB)
            target_volume = np.power(10.0, target_volume_dB / 20.0)
        elif self.volume_min_sample and self.volume_max_sample:
            target_volume = np.random.uniform(self.volume_min_sample, self.volume_max_sample)

        if self.use_rms_volume:
            noisy_volume = np.sqrt(np.mean(self.vad_merge(noisy_data) ** 2) + 1e-8)
            clean_volume = np.sqrt(np.mean(self.vad_merge(clean_data) ** 2) + 1e-8)
        else:
            noisy_volume = np.abs(noisy_data).max()
            clean_volume = np.abs(clean_data).max()

        noisy_volume = max(noisy_volume, clean_volume)
        scale = target_volume / (noisy_volume + 1e-6)
        noisy_data *= scale
        clean_data *= scale

        return noisy_data, clean_data, target_volume

    @staticmethod
    def volume_clip_dual(noisy_data, clean_data):
        noisy_volume = np.abs(noisy_data).max()
        clean_volume = np.abs(clean_data).max()
        noisy_volume = max(noisy_volume, clean_volume)
        if noisy_volume > 0.99:
            noisy_data *= 0.99 / noisy_volume
            clean_data *= 0.99 / noisy_volume
        return noisy_data, clean_data


def configure_dataset(debug=True):
    dataset = Dataset(
        # clean and noise path
        # clean_list_path="/root/data/lists/mix_voicebank_HQTTS_valid.list",
        # noise_list_path="/root/data/lists/freesound_download_audio_10to60s_filtered_20220819.lists",
        # clean_json_path="/data2/zhounan/data/lists/aishell3_bibletts_hqtts_vctk_48k_total_shuf_valid.json",
        # noise_json_path="/data2/zhounan/data/lists/freesound_download_audio_10to60s_filtered_20220819_selected_for_ssi.json",
        # clean_json_path="/root/autodl-tmp/data/lists/Aishell3_biaobei1w_BibbleTTS_HQTTS_VCTK_48k_wavs_shuf_train.json",
        # noise_json_path="/root/autodl-tmp/data/lists/freesound_selected_scenes_addwindnoise_48k_wavs_shuf.json",
        clean_json_path="/data2/zhounan/data/lists/Aishell3_biaobei1w_BibbleTTS_HQTTS_VCTK_48k_wavs_shuf_train.json",
        noise_json_path="/data2/zhounan/data/lists/freesound_selected_scenes_addwindnoise_48k_wavs_shuf.json",
        check_list_files=False,
        # number of speakers
        min_n_speakers=1,
        max_n_speakers=1,
        # whether use duration filter
        min_duration_seconds=1,
        # whether remove dc offset
        remove_dc_offset=True,
        # target sampling rate
        sampling_rate=48000,
        resample_method="soxr_vhq",
        # speech splice configuration
        speech_splice=True,
        speech_splice_equal_volume=True,
        speech_splice_equal_volume_range=[-6, 6],
        speech_splice_seconds=8,
        speech_random_start=False,
        add_extra_space_prob=0.3,
        # reverbrate configuration
        reverb_prob=0.5,
        # reverb_prob=0.5,
        reverb_use_FRA=False,
        # rir_list_path="/root/data/lists/rirs_monaural_farfield_20231008_48k_alldistances.list",
        rir_list_path="/data2/zhounan/data/lists/rirs_monaural_farfield_20231108_48k_alldistances.list",
        # rir_list_path="/root/autodl-tmp/data/lists/rirs_monaural_farfield_20231008_48k_alldistances.list",
        reverb_noise=False,
        # min_rt60=0.1,
        # max_rt60=0.8,
        # add noise configuration
        add_noise_prob=0.5,
        # add_noise_prob=0.9,
        only_noise_prob=0,
        noise_repeat_splice=True,
        trim_noise=True,
        snr_min=10,
        snr_max=30,
        noise_mix_prob=0.5,
        # speed perturb configuration
        speed_perturb_prob=0,
        speed_rate_min=0.8,
        speed_rate_max=1.2,
        # pitch perturb configuration
        pitch_shift_prob=0,
        semitones_down=-1.5,
        semitones_up=1.5,
        # loudness perturb configuration
        loudness_perturb_prob=0,
        loudness_min_factor=0.1,
        loudness_max_factor=10,
        loudness_max_n_intervals=5,
        # hard clipping perturb configuration
        clip_prob=0.2,
        # clip_prob=0,
        hard_clip_portion=1,
        hard_clip_on_rate=True,
        hard_clip_rate_min=0,
        hard_clip_rate_max=0.2,
        # sorf clipping perturb configuration
        soft_clip_types=["sox", "pedal", "soft", "sigmoid1", "sigmoid2"],
        # eq perturb configuration
        eq_perturb_prob=0.2,
        eq_db_min=-15,
        eq_db_max=5,
        # eq much gain perturb configuration
        eq_much_gain_prob=0.1,
        eq_much_gain_db_min=5,
        eq_much_gain_db_max=20,
        eq_much_gain_freq_min=1500,
        eq_much_gain_freq_max=12000,
        # band reject perturb configuration
        band_reject_prob=0.15,
        # band_reject_prob=0,
        band_reject_min_center_freq=100,
        band_reject_max_center_freq=16000,
        # band_reject_min_q=0.5,
        # band_reject_max_q=8,
        band_reject_min_freq_bandwidth=20,
        band_reject_max_freq_bandwidth=500,
        band_reject_use_stft=True,
        band_reject_max_n=2,
        # bass boost perturb configuration
        bass_boost_prob=0.13,
        bass_boost_highpass_cutoff_min=500,
        bass_boost_highpass_cutoff_max=2000,
        bass_boost_attenuation_min_db=-25,
        # DC offset perturb configuration
        dc_offset_prob=0.1,
        # dc_offset_prob=0,
        dc_offset_min=0.001,
        dc_offset_max=0.2,
        # spectral leakage perturb configuration
        spectral_leakage_prob=0.05,
        spectral_leakage_window_lengths=[1024, 2048, 4096],
        spectral_leakage_max_time_shift=20,
        # colored noise perturb configuration
        colored_noise_prob=0.5,
        # colored_noise_prob=0,
        colered_noise_snr_min=10,
        colered_noise_snr_max=50,
        colered_noise_types=["white", "pink", "equalized"],
        # low pass perturb configuration
        lowpass_prob=0.6,
        # lowpass_prob=0,
        lowpass_min_cutoff_freq=900,
        lowpass_max_cutoff_freq=15000,
        lowpass_min_order=4,
        lowpass_max_order=20,
        # spectra time frequency hole mask
        spectral_time_freq_holes_prob=0.15,
        spectral_time_freq_holes_stft_frame_length=1024,
        spectral_time_freq_holes_stft_frame_step=256,
        spectral_time_freq_holes_stft_holes_num_min=1,
        spectral_time_freq_holes_stft_holes_num_max=150,
        spectral_time_freq_holes_stft_holes_width_min_freq=1,
        spectral_time_freq_holes_stft_holes_width_max_freq=12,
        spectral_time_freq_holes_stft_holes_width_min_time=1,
        spectral_time_freq_holes_stft_holes_width_max_time=12,
        spectral_time_freq_holes_cutoff_freq=10000,
        # webrtc ns configuration
        webrtc_ns_prob=0,
        webrtc_ns_levels=[0, 1, 2, 3],
        webrtc_ns_volume_protection=True,
        # webrtc agc configuration
        webrtc_agc_prob=0,
        webrtc_agc_target_level_dbfs_max=-3,
        webrtc_agc_target_level_dbfs_min=-31,
        # drc configuration
        drc_prob=0,
        drc_threshold_db_min=-50,
        drc_threshold_db_max=0,
        drc_ratio_min=1,
        drc_ratio_max=20,
        drc_attack_ms_min=0.5,
        drc_attack_ms_max=5.0,
        drc_release_ms_min=50,
        drc_release_ms_max=1000,
        # codecs perturb configuration
        codecs_prob=0.3,
        # codecs_types=["mp3", "aac", "gsm", "opus"],
        codecs_types=["mp3", "gsm", "opus"],
        # packet loss perturb configuration
        packet_loss_prob=0.3,
        packet_loss_rate_min=0.05,
        packet_loss_rate_max=0.15,
        packet_loss_frame_time_min=0.008,
        packet_loss_frame_time_max=0.05,
        packet_loss_decay_rate_min=0,
        packet_loss_decay_rate_max=0.2,
        packet_loss_hard_loss_prob=1.0,
        packet_loss_on_vad=False,
        # bit crush perturb configuration
        bit_crush_prob=0,
        bit_crush_bit_min=4,
        bit_crush_bit_max=32,
        # colored noise post perturb configuration
        colored_noise_post_prob=0.1,
        # colored_noise_post_prob=0,
        colored_noise_post_snr_min=10,
        colored_noise_post_snr_max=50,
        colored_noise_post_types=["white", "pink", "equalized"],
        # random volume configuration
        random_volume=True,
        # volume_min_dB=-40,
        # volume_max_dB=-2,
        volume_min_sample=500.0 / 32768.0,
        volume_max_sample=0.99,
        use_rms_volume=False,
        sync_random_volume=True,
        # output random cut
        # output_cut_seconds=None,
        # output_random_cut=True,
        # output_normalize=True,
        output_resample=False,
        output_resample_rate=48000,
        debug=debug,
    )
    return dataset


def configure_dataset_24k(debug=True):
    dataset = Dataset(
        clean_json_path="/root/data/lists/speech_LibriTTS_R_24k_tail100.json",
        noise_json_path="/root/data/lists/noise_Demand_Office_48k.json",
        check_list_files=False,
        # number of speakers
        min_n_speakers=1,
        max_n_speakers=1,
        # whether use duration filter
        min_duration_seconds=1,
        # whether remove dc offset
        remove_dc_offset=True,
        # target sampling rate
        sampling_rate=24000,
        resample_method="fft",
        # speech splice configuration
        speech_splice=True,
        speech_splice_equal_volume=True,
        speech_splice_equal_volume_range=[-6, 6],
        speech_splice_seconds=6,
        speech_random_start=False,
        add_extra_space_prob=0.3,
        # reverbrate configuration
        reverb_prob=0.5,
        # reverb_prob=0.5,
        reverb_use_FRA=False,
        # rir_list_path="/root/data/lists/rirs_monaural_farfield_20231008_48k_alldistances.list",
        rir_list_path="/root/data/lists/rirs_monaural_farfield_20231108_24k_alldistances.list",
        # rir_list_path="/root/autodl-tmp/data/lists/rirs_monaural_farfield_20231008_48k_alldistances.list",
        reverb_noise=False,
        # min_rt60=0.1,
        # max_rt60=0.8,
        # add noise configuration
        add_noise_prob=0.5,
        # add_noise_prob=0.9,
        only_noise_prob=0,
        noise_repeat_splice=True,
        trim_noise=True,
        snr_min=10,
        snr_max=30,
        noise_mix_prob=0.5,
        # speed perturb configuration
        speed_perturb_prob=0,
        speed_rate_min=0.8,
        speed_rate_max=1.2,
        # pitch perturb configuration
        pitch_shift_prob=0,
        semitones_down=-1.5,
        semitones_up=1.5,
        # loudness perturb configuration
        loudness_perturb_prob=0,
        loudness_min_factor=0.1,
        loudness_max_factor=10,
        loudness_max_n_intervals=5,
        # hard clipping perturb configuration
        clip_prob=0.2,
        # clip_prob=0,
        hard_clip_portion=1,
        hard_clip_on_rate=True,
        hard_clip_rate_min=0,
        hard_clip_rate_max=0.2,
        # sorf clipping perturb configuration
        soft_clip_types=["sox", "pedal", "soft", "sigmoid1", "sigmoid2"],
        # eq perturb configuration
        eq_perturb_prob=0.2,
        eq_db_min=-15,
        eq_db_max=5,
        # eq much gain perturb configuration
        eq_much_gain_prob=0.1,
        eq_much_gain_db_min=5,
        eq_much_gain_db_max=20,
        eq_much_gain_freq_min=1500,
        eq_much_gain_freq_max=12000,
        # band reject perturb configuration
        band_reject_prob=0.15,
        # band_reject_prob=0,
        band_reject_min_center_freq=100,
        band_reject_max_center_freq=12000,
        # band_reject_min_q=0.5,
        # band_reject_max_q=8,
        band_reject_min_freq_bandwidth=20,
        band_reject_max_freq_bandwidth=500,
        band_reject_use_stft=True,
        band_reject_max_n=2,
        # bass boost perturb configuration
        bass_boost_prob=0.13,
        bass_boost_highpass_cutoff_min=500,
        bass_boost_highpass_cutoff_max=2000,
        bass_boost_attenuation_min_db=-25,
        # DC offset perturb configuration
        dc_offset_prob=0.1,
        # dc_offset_prob=0,
        dc_offset_min=0.001,
        dc_offset_max=0.2,
        # spectral leakage perturb configuration
        spectral_leakage_prob=0.05,
        spectral_leakage_window_lengths=[1024, 2048, 4096],
        spectral_leakage_max_time_shift=20,
        # colored noise perturb configuration
        colored_noise_prob=0.5,
        # colored_noise_prob=0,
        colered_noise_snr_min=10,
        colered_noise_snr_max=50,
        colered_noise_types=["white", "pink", "equalized"],
        # low pass perturb configuration
        lowpass_prob=0.6,
        # lowpass_prob=0,
        lowpass_min_cutoff_freq=900,
        lowpass_max_cutoff_freq=12000,
        lowpass_min_order=4,
        lowpass_max_order=20,
        # spectra time frequency hole mask
        spectral_time_freq_holes_prob=0.15,
        spectral_time_freq_holes_stft_frame_length=1024,
        spectral_time_freq_holes_stft_frame_step=256,
        spectral_time_freq_holes_stft_holes_num_min=1,
        spectral_time_freq_holes_stft_holes_num_max=150,
        spectral_time_freq_holes_stft_holes_width_min_freq=1,
        spectral_time_freq_holes_stft_holes_width_max_freq=12,
        spectral_time_freq_holes_stft_holes_width_min_time=1,
        spectral_time_freq_holes_stft_holes_width_max_time=12,
        spectral_time_freq_holes_cutoff_freq=10000,
        # webrtc ns configuration
        webrtc_ns_prob=0,
        webrtc_ns_levels=[0, 1, 2, 3],
        webrtc_ns_volume_protection=True,
        # webrtc agc configuration
        webrtc_agc_prob=0,
        webrtc_agc_target_level_dbfs_max=-3,
        webrtc_agc_target_level_dbfs_min=-31,
        # drc configuration
        drc_prob=0,
        drc_threshold_db_min=-50,
        drc_threshold_db_max=0,
        drc_ratio_min=1,
        drc_ratio_max=20,
        drc_attack_ms_min=0.5,
        drc_attack_ms_max=5.0,
        drc_release_ms_min=50,
        drc_release_ms_max=1000,
        # codecs perturb configuration
        codecs_prob=0.3,
        # codecs_types=["mp3", "aac", "gsm", "opus"],
        codecs_types=["mp3", "gsm"],
        # packet loss perturb configuration
        packet_loss_prob=0.3,
        packet_loss_rate_min=0.05,
        packet_loss_rate_max=0.15,
        packet_loss_frame_time_min=0.008,
        packet_loss_frame_time_max=0.05,
        packet_loss_decay_rate_min=0,
        packet_loss_decay_rate_max=0.2,
        packet_loss_hard_loss_prob=1.0,
        packet_loss_on_vad=False,
        # bit crush perturb configuration
        bit_crush_prob=0,
        bit_crush_bit_min=4,
        bit_crush_bit_max=32,
        # colored noise post perturb configuration
        colored_noise_post_prob=0.1,
        # colored_noise_post_prob=0,
        colored_noise_post_snr_min=10,
        colored_noise_post_snr_max=50,
        colored_noise_post_types=["white", "pink", "equalized"],
        # random volume configuration
        random_volume=True,
        # volume_min_dB=-40,
        # volume_max_dB=-2,
        volume_min_sample=500.0 / 32768.0,
        volume_max_sample=0.99,
        use_rms_volume=False,
        sync_random_volume=True,
        # output random cut
        # output_cut_seconds=None,
        # output_random_cut=True,
        # output_normalize=True,
        output_normalize=True,
        output_resample=False,
        output_resample_rate=24000,
        debug=debug,
    )
    return dataset


# @profile
def test_dataset_and_memory_profile():
    import os
    import subprocess

    from torch.utils.data import DataLoader
    from tqdm import tqdm

    # import tracemalloc

    np.random.seed(8886)
    save_path = "./test_case_comm_distort_simu_dataset"
    subprocess.run(["rm", "-rf", f"{save_path}"])
    os.makedirs(save_path, exist_ok=True)

    # tracemalloc.start()
    # snapshot = tracemalloc.take_snapshot()
    dataset = configure_dataset_24k(debug=True)

    print("dataset length:", len(dataset))
    test_num = 100
    for i in tqdm(range(test_num)):
        #  # 获取新快照
        # snapshot_new = tracemalloc.take_snapshot()

        # # 统计新增的内存分配
        # top_stats = snapshot_new.compare_to(snapshot, 'lineno')

        # print("[ Top 10 new allocations ]")
        # for stat in top_stats[:10]:
        #     print(stat)

        # # 保存当前快照作为下次的旧快照
        # snapshot = snapshot_new
        random_index = np.random.randint(0, len(dataset))
        data = dataset.__getitem__(random_index)
        step = 0
        for key in data.keys():
            if type(data[key]) == np.ndarray:
                savename = os.path.join(save_path, f"{i}_step{step}_{key}.wav")
                if key != "perturbed":
                    sf.write(savename, data[key], dataset.sampling_rate)
                else:
                    sf.write(savename, data[key], dataset.output_resample_rate)
                step += 1

    #  # 获取新快照
    # snapshot_new = tracemalloc.take_snapshot()

    # # 统计新增的内存分配
    # top_stats = snapshot_new.compare_to(snapshot, 'lineno')

    # print("[ Top 10 new allocations ]")
    # for stat in top_stats[:10]:
    #     print(stat)

    # # 保存当前快照作为下次的旧快照
    # snapshot = snapshot_new


if __name__ == "__main__":
    test_dataset_and_memory_profile()
    # test_dataloader_and_memory_profile()
