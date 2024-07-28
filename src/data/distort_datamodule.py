from typing import Any, Dict, Optional

from lightning import LightningDataModule
from torch.utils.data import DataLoader

from .components.collate import pad_to_longest_monaural
from .components.comm_distort_simu_dataset import Dataset


class DistortDataModule(LightningDataModule):
    """`LightningDataModule` for the distortion dataset.
    """

    def __init__(
        self,
        # clean and noise path
        clean_list_path_train=None,
        clean_json_path_train=None,
        noise_list_path_train=None,
        noise_json_path_train=None,
        clean_list_path_valid=None,
        clean_json_path_valid=None,
        noise_list_path_valid=None,
        noise_json_path_valid=None,
        clean_list_path_test=None,
        clean_json_path_test=None,
        noise_list_path_test=None,
        noise_json_path_test=None,
        rir_list_path_train=None,
        rir_list_path_valid=None,
        rir_list_path_test=None,
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
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        prefetch_factor: int = 0,
    ) -> None:
        """Initialize a `MNISTDataModule`.

        :param data_dir: The data directory. Defaults to `"data/"`.
        :param train_val_test_split: The train, validation and test split. Defaults to `(55_000, 5_000, 10_000)`.
        :param batch_size: The batch size. Defaults to `64`.
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data_train = Dataset(
            clean_list_path=clean_list_path_train,
            clean_json_path=clean_json_path_train,
            noise_list_path=noise_list_path_train,
            noise_json_path=noise_json_path_train,
            check_list_files=check_list_files,
            min_n_speakers=min_n_speakers,
            max_n_speakers=max_n_speakers,
            min_duration_seconds=min_duration_seconds,
            max_duration_seconds=max_duration_seconds,
            remove_dc_offset=remove_dc_offset,
            sampling_rate=sampling_rate,
            resample_method=resample_method,
            speech_splice=speech_splice,
            speech_splice_equal_volume=speech_splice_equal_volume,
            speech_splice_equal_volume_range=speech_splice_equal_volume_range,
            speech_splice_seconds=speech_splice_seconds,
            speech_random_start=speech_random_start,
            add_extra_space_prob=add_extra_space_prob,
            reverb_prob=reverb_prob,
            reverb_use_FRA=reverb_use_FRA,
            rir_list_path=rir_list_path_train,
            reverb_noise=reverb_noise,
            min_rt60=min_rt60,
            max_rt60=max_rt60,
            add_noise_prob=add_noise_prob,
            only_noise_prob=only_noise_prob,
            noise_repeat_splice=noise_repeat_splice,
            trim_noise=trim_noise,
            snr_min=snr_min,
            snr_max=snr_max,
            noise_mix_prob=noise_mix_prob,
            speed_perturb_prob=speed_perturb_prob,
            speed_rate_min=speed_rate_min,
            speed_rate_max=speed_rate_max,
            pitch_shift_prob=pitch_shift_prob,
            semitones_down=semitones_down,
            semitones_up=semitones_up,
            loudness_perturb_prob=loudness_perturb_prob,
            loudness_min_factor=loudness_min_factor,
            loudness_max_factor=loudness_max_factor,
            loudness_max_n_intervals=loudness_max_n_intervals,
            clip_prob=clip_prob,
            hard_clip_portion=hard_clip_portion,
            hard_clip_on_rate=hard_clip_on_rate,
            hard_clip_rate_min=hard_clip_rate_min,
            hard_clip_rate_max=hard_clip_rate_max,
            hard_clip_threshold_db_min=hard_clip_threshold_db_min,
            hard_clip_threshold_db_max=hard_clip_threshold_db_max,
            soft_clip_types=soft_clip_types,
            eq_perturb_prob=eq_perturb_prob,
            eq_db_min=eq_db_min,
            eq_db_max=eq_db_max,
            eq_much_gain_prob=eq_much_gain_prob,
            eq_much_gain_db_min=eq_much_gain_db_min,
            eq_much_gain_db_max=eq_much_gain_db_max,
            eq_much_gain_freq_min=eq_much_gain_freq_min,
            eq_much_gain_freq_max=eq_much_gain_freq_max,
            band_reject_prob=band_reject_prob,
            band_reject_min_center_freq=band_reject_min_center_freq,
            band_reject_max_center_freq=band_reject_max_center_freq,
            band_reject_min_q=band_reject_min_q,
            band_reject_max_q=band_reject_max_q,
            band_reject_min_freq_bandwidth=band_reject_min_freq_bandwidth,
            band_reject_max_freq_bandwidth=band_reject_max_freq_bandwidth,
            band_reject_use_stft=band_reject_use_stft,
            band_reject_max_n=band_reject_max_n,
            bass_boost_prob=bass_boost_prob,
            bass_boost_highpass_cutoff_min=bass_boost_highpass_cutoff_min,
            bass_boost_highpass_cutoff_max=bass_boost_highpass_cutoff_max,
            bass_boost_attenuation_min_db=bass_boost_attenuation_min_db,
            dc_offset_prob=dc_offset_prob,
            dc_offset_min=dc_offset_min,
            dc_offset_max=dc_offset_max,
            spectral_leakage_prob=spectral_leakage_prob,
            spectral_leakage_window_lengths=spectral_leakage_window_lengths,
            spectral_leakage_max_time_shift=spectral_leakage_max_time_shift,
            colored_noise_prob=colored_noise_prob,
            colered_noise_snr_min=colered_noise_snr_min,
            colered_noise_snr_max=colered_noise_snr_max,
            colered_noise_types=colered_noise_types,
            lowpass_prob=lowpass_prob,
            lowpass_min_cutoff_freq=lowpass_min_cutoff_freq,
            lowpass_max_cutoff_freq=lowpass_max_cutoff_freq,
            lowpass_min_order=lowpass_min_order,
            lowpass_max_order=lowpass_max_order,
            spectral_time_freq_holes_prob=spectral_time_freq_holes_prob,
            spectral_time_freq_holes_stft_frame_length=spectral_time_freq_holes_stft_frame_length,
            spectral_time_freq_holes_stft_frame_step=spectral_time_freq_holes_stft_frame_step,
            spectral_time_freq_holes_stft_holes_num_min=spectral_time_freq_holes_stft_holes_num_min,
            spectral_time_freq_holes_stft_holes_num_max=spectral_time_freq_holes_stft_holes_num_max,
            spectral_time_freq_holes_stft_holes_width_min_freq=spectral_time_freq_holes_stft_holes_width_min_freq,
            spectral_time_freq_holes_stft_holes_width_max_freq=spectral_time_freq_holes_stft_holes_width_max_freq,
            spectral_time_freq_holes_stft_holes_width_min_time=spectral_time_freq_holes_stft_holes_width_min_time,
            spectral_time_freq_holes_stft_holes_width_max_time=spectral_time_freq_holes_stft_holes_width_max_time,
            spectral_time_freq_holes_cutoff_freq=spectral_time_freq_holes_cutoff_freq,
            webrtc_ns_prob=webrtc_ns_prob,
            webrtc_ns_levels=webrtc_ns_levels,
            webrtc_ns_volume_protection=webrtc_ns_volume_protection,
            webrtc_agc_prob=webrtc_agc_prob,
            webrtc_agc_target_level_dbfs_max=webrtc_agc_target_level_dbfs_max,
            webrtc_agc_target_level_dbfs_min=webrtc_agc_target_level_dbfs_min,
            drc_prob=drc_prob,
            drc_threshold_db_min=drc_threshold_db_min,
            drc_threshold_db_max=drc_threshold_db_max,
            drc_ratio_min=drc_ratio_min,
            drc_ratio_max=drc_ratio_max,
            drc_attack_ms_min=drc_attack_ms_min,
            drc_attack_ms_max=drc_attack_ms_max,
            drc_release_ms_min=drc_release_ms_min,
            drc_release_ms_max=drc_release_ms_max,
            codecs_prob=codecs_prob,
            codecs_types=codecs_types,
            packet_loss_prob=packet_loss_prob,
            packet_loss_rate_min=packet_loss_rate_min,
            packet_loss_rate_max=packet_loss_rate_max,
            packet_loss_frame_time_min=packet_loss_frame_time_min,
            packet_loss_frame_time_max=packet_loss_frame_time_max,
            packet_loss_decay_rate_min=packet_loss_decay_rate_min,
            packet_loss_decay_rate_max=packet_loss_decay_rate_max,
            packet_loss_hard_loss_prob=packet_loss_hard_loss_prob,
            packet_loss_on_vad=packet_loss_on_vad,
            bit_crush_prob=bit_crush_prob,
            bit_crush_bit_min=bit_crush_bit_min,
            bit_crush_bit_max=bit_crush_bit_max,
            colored_noise_post_prob=colored_noise_post_prob,
            colored_noise_post_snr_min=colored_noise_post_snr_min,
            colored_noise_post_snr_max=colored_noise_post_snr_max,
            colored_noise_post_types=colored_noise_post_types,
            random_volume=random_volume,
            volume_min_dB=volume_min_dB,
            volume_max_dB=volume_max_dB,
            volume_min_sample=volume_min_sample,
            volume_max_sample=volume_max_sample,
            use_rms_volume=use_rms_volume,
            sync_random_volume=sync_random_volume,
            output_cut_seconds=output_cut_seconds,
            output_random_cut=output_random_cut,
            output_normalize=output_normalize,
            output_resample=output_resample,
            output_resample_rate=output_resample_rate,
            debug=debug,
            dummy=dummy,
        )

        self.data_val = Dataset(
            clean_list_path=clean_list_path_valid,
            clean_json_path=clean_json_path_valid,
            noise_list_path=noise_list_path_valid,
            noise_json_path=noise_json_path_valid,
            check_list_files=check_list_files,
            min_n_speakers=min_n_speakers,
            max_n_speakers=max_n_speakers,
            min_duration_seconds=min_duration_seconds,
            max_duration_seconds=max_duration_seconds,
            remove_dc_offset=remove_dc_offset,
            sampling_rate=sampling_rate,
            resample_method=resample_method,
            speech_splice=speech_splice,
            speech_splice_equal_volume=speech_splice_equal_volume,
            speech_splice_equal_volume_range=speech_splice_equal_volume_range,
            speech_splice_seconds=speech_splice_seconds,
            speech_random_start=speech_random_start,
            add_extra_space_prob=add_extra_space_prob,
            reverb_prob=reverb_prob,
            reverb_use_FRA=reverb_use_FRA,
            rir_list_path=rir_list_path_valid,
            reverb_noise=reverb_noise,
            min_rt60=min_rt60,
            max_rt60=max_rt60,
            add_noise_prob=add_noise_prob,
            only_noise_prob=only_noise_prob,
            noise_repeat_splice=noise_repeat_splice,
            trim_noise=trim_noise,
            snr_min=snr_min,
            snr_max=snr_max,
            noise_mix_prob=noise_mix_prob,
            speed_perturb_prob=speed_perturb_prob,
            speed_rate_min=speed_rate_min,
            speed_rate_max=speed_rate_max,
            pitch_shift_prob=pitch_shift_prob,
            semitones_down=semitones_down,
            semitones_up=semitones_up,
            loudness_perturb_prob=loudness_perturb_prob,
            loudness_min_factor=loudness_min_factor,
            loudness_max_factor=loudness_max_factor,
            loudness_max_n_intervals=loudness_max_n_intervals,
            clip_prob=clip_prob,
            hard_clip_portion=hard_clip_portion,
            hard_clip_on_rate=hard_clip_on_rate,
            hard_clip_rate_min=hard_clip_rate_min,
            hard_clip_rate_max=hard_clip_rate_max,
            hard_clip_threshold_db_min=hard_clip_threshold_db_min,
            hard_clip_threshold_db_max=hard_clip_threshold_db_max,
            soft_clip_types=soft_clip_types,
            eq_perturb_prob=eq_perturb_prob,
            eq_db_min=eq_db_min,
            eq_db_max=eq_db_max,
            eq_much_gain_prob=eq_much_gain_prob,
            eq_much_gain_db_min=eq_much_gain_db_min,
            eq_much_gain_db_max=eq_much_gain_db_max,
            eq_much_gain_freq_min=eq_much_gain_freq_min,
            eq_much_gain_freq_max=eq_much_gain_freq_max,
            band_reject_prob=band_reject_prob,
            band_reject_min_center_freq=band_reject_min_center_freq,
            band_reject_max_center_freq=band_reject_max_center_freq,
            band_reject_min_q=band_reject_min_q,
            band_reject_max_q=band_reject_max_q,
            band_reject_min_freq_bandwidth=band_reject_min_freq_bandwidth,
            band_reject_max_freq_bandwidth=band_reject_max_freq_bandwidth,
            band_reject_use_stft=band_reject_use_stft,
            band_reject_max_n=band_reject_max_n,
            bass_boost_prob=bass_boost_prob,
            bass_boost_highpass_cutoff_min=bass_boost_highpass_cutoff_min,
            bass_boost_highpass_cutoff_max=bass_boost_highpass_cutoff_max,
            bass_boost_attenuation_min_db=bass_boost_attenuation_min_db,
            dc_offset_prob=dc_offset_prob,
            dc_offset_min=dc_offset_min,
            dc_offset_max=dc_offset_max,
            spectral_leakage_prob=spectral_leakage_prob,
            spectral_leakage_window_lengths=spectral_leakage_window_lengths,
            spectral_leakage_max_time_shift=spectral_leakage_max_time_shift,
            colored_noise_prob=colored_noise_prob,
            colered_noise_snr_min=colered_noise_snr_min,
            colered_noise_snr_max=colered_noise_snr_max,
            colered_noise_types=colered_noise_types,
            lowpass_prob=lowpass_prob,
            lowpass_min_cutoff_freq=lowpass_min_cutoff_freq,
            lowpass_max_cutoff_freq=lowpass_max_cutoff_freq,
            lowpass_min_order=lowpass_min_order,
            lowpass_max_order=lowpass_max_order,
            spectral_time_freq_holes_prob=spectral_time_freq_holes_prob,
            spectral_time_freq_holes_stft_frame_length=spectral_time_freq_holes_stft_frame_length,
            spectral_time_freq_holes_stft_frame_step=spectral_time_freq_holes_stft_frame_step,
            spectral_time_freq_holes_stft_holes_num_min=spectral_time_freq_holes_stft_holes_num_min,
            spectral_time_freq_holes_stft_holes_num_max=spectral_time_freq_holes_stft_holes_num_max,
            spectral_time_freq_holes_stft_holes_width_min_freq=spectral_time_freq_holes_stft_holes_width_min_freq,
            spectral_time_freq_holes_stft_holes_width_max_freq=spectral_time_freq_holes_stft_holes_width_max_freq,
            spectral_time_freq_holes_stft_holes_width_min_time=spectral_time_freq_holes_stft_holes_width_min_time,
            spectral_time_freq_holes_stft_holes_width_max_time=spectral_time_freq_holes_stft_holes_width_max_time,
            spectral_time_freq_holes_cutoff_freq=spectral_time_freq_holes_cutoff_freq,
            webrtc_ns_prob=webrtc_ns_prob,
            webrtc_ns_levels=webrtc_ns_levels,
            webrtc_ns_volume_protection=webrtc_ns_volume_protection,
            webrtc_agc_prob=webrtc_agc_prob,
            webrtc_agc_target_level_dbfs_max=webrtc_agc_target_level_dbfs_max,
            webrtc_agc_target_level_dbfs_min=webrtc_agc_target_level_dbfs_min,
            drc_prob=drc_prob,
            drc_threshold_db_min=drc_threshold_db_min,
            drc_threshold_db_max=drc_threshold_db_max,
            drc_ratio_min=drc_ratio_min,
            drc_ratio_max=drc_ratio_max,
            drc_attack_ms_min=drc_attack_ms_min,
            drc_attack_ms_max=drc_attack_ms_max,
            drc_release_ms_min=drc_release_ms_min,
            drc_release_ms_max=drc_release_ms_max,
            codecs_prob=codecs_prob,
            codecs_types=codecs_types,
            packet_loss_prob=packet_loss_prob,
            packet_loss_rate_min=packet_loss_rate_min,
            packet_loss_rate_max=packet_loss_rate_max,
            packet_loss_frame_time_min=packet_loss_frame_time_min,
            packet_loss_frame_time_max=packet_loss_frame_time_max,
            packet_loss_decay_rate_min=packet_loss_decay_rate_min,
            packet_loss_decay_rate_max=packet_loss_decay_rate_max,
            packet_loss_hard_loss_prob=packet_loss_hard_loss_prob,
            packet_loss_on_vad=packet_loss_on_vad,
            bit_crush_prob=bit_crush_prob,
            bit_crush_bit_min=bit_crush_bit_min,
            bit_crush_bit_max=bit_crush_bit_max,
            colored_noise_post_prob=colored_noise_post_prob,
            colored_noise_post_snr_min=colored_noise_post_snr_min,
            colored_noise_post_snr_max=colored_noise_post_snr_max,
            colored_noise_post_types=colored_noise_post_types,
            random_volume=random_volume,
            volume_min_dB=volume_min_dB,
            volume_max_dB=volume_max_dB,
            volume_min_sample=volume_min_sample,
            volume_max_sample=volume_max_sample,
            use_rms_volume=use_rms_volume,
            sync_random_volume=sync_random_volume,
            output_cut_seconds=output_cut_seconds,
            output_random_cut=output_random_cut,
            output_normalize=output_normalize,
            output_resample=output_resample,
            output_resample_rate=output_resample_rate,
            debug=debug,
            dummy=dummy,
        )

        self.data_test = Dataset(
            clean_list_path=clean_list_path_test,
            clean_json_path=clean_json_path_test,
            noise_list_path=noise_list_path_test,
            noise_json_path=noise_json_path_test,
            check_list_files=check_list_files,
            min_n_speakers=min_n_speakers,
            max_n_speakers=max_n_speakers,
            min_duration_seconds=min_duration_seconds,
            max_duration_seconds=max_duration_seconds,
            remove_dc_offset=remove_dc_offset,
            sampling_rate=sampling_rate,
            resample_method=resample_method,
            speech_splice=speech_splice,
            speech_splice_equal_volume=speech_splice_equal_volume,
            speech_splice_equal_volume_range=speech_splice_equal_volume_range,
            speech_splice_seconds=speech_splice_seconds,
            speech_random_start=speech_random_start,
            add_extra_space_prob=add_extra_space_prob,
            reverb_prob=reverb_prob,
            reverb_use_FRA=reverb_use_FRA,
            rir_list_path=rir_list_path_test,
            reverb_noise=reverb_noise,
            min_rt60=min_rt60,
            max_rt60=max_rt60,
            add_noise_prob=add_noise_prob,
            only_noise_prob=only_noise_prob,
            noise_repeat_splice=noise_repeat_splice,
            trim_noise=trim_noise,
            snr_min=snr_min,
            snr_max=snr_max,
            noise_mix_prob=noise_mix_prob,
            speed_perturb_prob=speed_perturb_prob,
            speed_rate_min=speed_rate_min,
            speed_rate_max=speed_rate_max,
            pitch_shift_prob=pitch_shift_prob,
            semitones_down=semitones_down,
            semitones_up=semitones_up,
            loudness_perturb_prob=loudness_perturb_prob,
            loudness_min_factor=loudness_min_factor,
            loudness_max_factor=loudness_max_factor,
            loudness_max_n_intervals=loudness_max_n_intervals,
            clip_prob=clip_prob,
            hard_clip_portion=hard_clip_portion,
            hard_clip_on_rate=hard_clip_on_rate,
            hard_clip_rate_min=hard_clip_rate_min,
            hard_clip_rate_max=hard_clip_rate_max,
            hard_clip_threshold_db_min=hard_clip_threshold_db_min,
            hard_clip_threshold_db_max=hard_clip_threshold_db_max,
            soft_clip_types=soft_clip_types,
            eq_perturb_prob=eq_perturb_prob,
            eq_db_min=eq_db_min,
            eq_db_max=eq_db_max,
            eq_much_gain_prob=eq_much_gain_prob,
            eq_much_gain_db_min=eq_much_gain_db_min,
            eq_much_gain_db_max=eq_much_gain_db_max,
            eq_much_gain_freq_min=eq_much_gain_freq_min,
            eq_much_gain_freq_max=eq_much_gain_freq_max,
            band_reject_prob=band_reject_prob,
            band_reject_min_center_freq=band_reject_min_center_freq,
            band_reject_max_center_freq=band_reject_max_center_freq,
            band_reject_min_q=band_reject_min_q,
            band_reject_max_q=band_reject_max_q,
            band_reject_min_freq_bandwidth=band_reject_min_freq_bandwidth,
            band_reject_max_freq_bandwidth=band_reject_max_freq_bandwidth,
            band_reject_use_stft=band_reject_use_stft,
            band_reject_max_n=band_reject_max_n,
            bass_boost_prob=bass_boost_prob,
            bass_boost_highpass_cutoff_min=bass_boost_highpass_cutoff_min,
            bass_boost_highpass_cutoff_max=bass_boost_highpass_cutoff_max,
            bass_boost_attenuation_min_db=bass_boost_attenuation_min_db,
            dc_offset_prob=dc_offset_prob,
            dc_offset_min=dc_offset_min,
            dc_offset_max=dc_offset_max,
            spectral_leakage_prob=spectral_leakage_prob,
            spectral_leakage_window_lengths=spectral_leakage_window_lengths,
            spectral_leakage_max_time_shift=spectral_leakage_max_time_shift,
            colored_noise_prob=colored_noise_prob,
            colered_noise_snr_min=colered_noise_snr_min,
            colered_noise_snr_max=colered_noise_snr_max,
            colered_noise_types=colered_noise_types,
            lowpass_prob=lowpass_prob,
            lowpass_min_cutoff_freq=lowpass_min_cutoff_freq,
            lowpass_max_cutoff_freq=lowpass_max_cutoff_freq,
            lowpass_min_order=lowpass_min_order,
            lowpass_max_order=lowpass_max_order,
            spectral_time_freq_holes_prob=spectral_time_freq_holes_prob,
            spectral_time_freq_holes_stft_frame_length=spectral_time_freq_holes_stft_frame_length,
            spectral_time_freq_holes_stft_frame_step=spectral_time_freq_holes_stft_frame_step,
            spectral_time_freq_holes_stft_holes_num_min=spectral_time_freq_holes_stft_holes_num_min,
            spectral_time_freq_holes_stft_holes_num_max=spectral_time_freq_holes_stft_holes_num_max,
            spectral_time_freq_holes_stft_holes_width_min_freq=spectral_time_freq_holes_stft_holes_width_min_freq,
            spectral_time_freq_holes_stft_holes_width_max_freq=spectral_time_freq_holes_stft_holes_width_max_freq,
            spectral_time_freq_holes_stft_holes_width_min_time=spectral_time_freq_holes_stft_holes_width_min_time,
            spectral_time_freq_holes_stft_holes_width_max_time=spectral_time_freq_holes_stft_holes_width_max_time,
            spectral_time_freq_holes_cutoff_freq=spectral_time_freq_holes_cutoff_freq,
            webrtc_ns_prob=webrtc_ns_prob,
            webrtc_ns_levels=webrtc_ns_levels,
            webrtc_ns_volume_protection=webrtc_ns_volume_protection,
            webrtc_agc_prob=webrtc_agc_prob,
            webrtc_agc_target_level_dbfs_max=webrtc_agc_target_level_dbfs_max,
            webrtc_agc_target_level_dbfs_min=webrtc_agc_target_level_dbfs_min,
            drc_prob=drc_prob,
            drc_threshold_db_min=drc_threshold_db_min,
            drc_threshold_db_max=drc_threshold_db_max,
            drc_ratio_min=drc_ratio_min,
            drc_ratio_max=drc_ratio_max,
            drc_attack_ms_min=drc_attack_ms_min,
            drc_attack_ms_max=drc_attack_ms_max,
            drc_release_ms_min=drc_release_ms_min,
            drc_release_ms_max=drc_release_ms_max,
            codecs_prob=codecs_prob,
            codecs_types=codecs_types,
            packet_loss_prob=packet_loss_prob,
            packet_loss_rate_min=packet_loss_rate_min,
            packet_loss_rate_max=packet_loss_rate_max,
            packet_loss_frame_time_min=packet_loss_frame_time_min,
            packet_loss_frame_time_max=packet_loss_frame_time_max,
            packet_loss_decay_rate_min=packet_loss_decay_rate_min,
            packet_loss_decay_rate_max=packet_loss_decay_rate_max,
            packet_loss_hard_loss_prob=packet_loss_hard_loss_prob,
            packet_loss_on_vad=packet_loss_on_vad,
            bit_crush_prob=bit_crush_prob,
            bit_crush_bit_min=bit_crush_bit_min,
            bit_crush_bit_max=bit_crush_bit_max,
            colored_noise_post_prob=colored_noise_post_prob,
            colored_noise_post_snr_min=colored_noise_post_snr_min,
            colored_noise_post_snr_max=colored_noise_post_snr_max,
            colored_noise_post_types=colored_noise_post_types,
            random_volume=random_volume,
            volume_min_dB=volume_min_dB,
            volume_max_dB=volume_max_dB,
            volume_min_sample=volume_min_sample,
            volume_max_sample=volume_max_sample,
            use_rms_volume=use_rms_volume,
            sync_random_volume=sync_random_volume,
            output_cut_seconds=output_cut_seconds,
            output_random_cut=output_random_cut,
            output_normalize=output_normalize,
            output_resample=output_resample,
            output_resample_rate=output_resample_rate,
            debug=debug,
            dummy=dummy,
        )

        self.batch_size_per_device = batch_size

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        # # load and split datasets only if not loaded already
        # if not self.data_train and not self.data_val and not self.data_test:
        #     trainset = MNIST(self.hparams.data_dir, train=True, transform=self.transforms)
        #     testset = MNIST(self.hparams.data_dir, train=False, transform=self.transforms)
        #     dataset = ConcatDataset(datasets=[trainset, testset])
        #     self.data_train, self.data_val, self.data_test = random_split(
        #         dataset=dataset,
        #         lengths=self.hparams.train_val_test_split,
        #         generator=torch.Generator().manual_seed(42),
        #     )

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=pad_to_longest_monaural,
            prefetch_factor=self.hparams.prefetch_factor,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=pad_to_longest_monaural,
            prefetch_factor=self.hparams.prefetch_factor,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=pad_to_longest_monaural,
            prefetch_factor=self.hparams.prefetch_factor,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass


if __name__ == "__main__":
    import os

    import hydra
    import soundfile as sf
    from omegaconf import DictConfig, OmegaConf

    temp_dir = "temp_data_folder"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    os.system(f"rm {temp_dir}/*")

    @hydra.main(version_base="1.3", config_path="../../configs/data", config_name="distort.yaml")
    def test_data_module(cfg: DictConfig) -> None:
        OmegaConf.set_struct(cfg, False)
        cfg.pop("_target_")
        data_module = DistortDataModule(**cfg)
        data_module.prepare_data()
        data_module.setup()
        train_dataloader = data_module.train_dataloader()
        for batch in train_dataloader:
            print(batch["perturbed"].shape)
            print(batch["clean"].shape)
            print(batch["name"])
            print(batch["sample_length"])
            print(batch["sampling_rate"])
            for i in range(batch["perturbed"].shape[0]):
                sf.write(
                    f"{temp_dir}/{batch['name'][i]}_perturbed.wav",
                    batch["perturbed"][i].numpy()[: batch["sample_length"][i]],
                    batch["sampling_rate"][i],
                )
                sf.write(
                    f"{temp_dir}/{batch['name'][i]}_clean.wav",
                    batch["clean"][i].numpy()[: batch["sample_length"][i]],
                    batch["sampling_rate"][i],
                )

    test_data_module()
