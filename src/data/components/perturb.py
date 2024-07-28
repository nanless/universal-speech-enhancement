import random
import subprocess

import ffmpeg
import librosa
import numpy as np
import opuslib
import pedalboard
import torch
import torchaudio.functional as taf
import webrtcvad
from numba import jit
from pedalboard import (
    Bitcrush,
    Clipping,
    Compressor,
    Distortion,
    GSMFullRateCompressor,
    Limiter,
    MP3Compressor,
    PitchShift,
)
from pysndfx import AudioEffectsChain
from scipy import signal


class SpeedPerturb:
    def __init__(self, sample_rate, min_speed_rate=0.8, max_speed_rate=1.2, speed_rate=None):
        self._sample_rate = sample_rate
        self._min_rate = min_speed_rate
        self._max_rate = max_speed_rate
        self._speed_rate = speed_rate

    def __call__(self, data):
        return self.process(data)

    def process(self, data):
        if self._speed_rate is not None:
            speed_rate = self._speed_rate
        else:
            speed_rate = np.random.uniform(self._min_rate, self._max_rate)
        if speed_rate <= 0:
            raise ValueError("speed_rate should be greater than zero.")
        # data = librosa.effects.time_stretch(data, rate=speed_rate)
        apply_audio_effects = AudioEffectsChain().tempo(speed_rate, opt_flag="s", segment=32)
        data = apply_audio_effects(src=data, sample_in=self._sample_rate)
        # data = torch.from_numpy(data)
        # speed_perturb = SpeedPerturbation(16000, [speed_rate])
        # data = speed_perturb(data)
        # data = data.numpy()
        return data, speed_rate


class PitchPerturb:
    def __init__(self, sample_rate, down_max_semitone=-1, up_max_semitone=1, semitone=None):
        self._sample_rate = sample_rate
        self._down_max_semitone = down_max_semitone
        self._up_max_semitone = up_max_semitone
        self._semitone = semitone

    def __call__(self, data):
        return self.process(data)

    def process(self, data):
        if self._semitone is not None:
            semitones = self._semitone
        else:
            semitones = np.random.uniform(self._down_max_semitone, self._up_max_semitone)
        # data = librosa.effects.pitch_shift(data, sr=self._sample_rate, n_steps=semitones)
        inst = PitchShift(semitones=semitones)
        data = inst(data, self._sample_rate)
        return data, semitones


# class EQPerturb:
#     def __init__(self, sample_rate, bandwidth_min=1, bandwidth_max=20, db_min=-10, db_max=0, db=None, num_bands_min=1, num_bands_max=10, num_bands=None):
#         self._sample_rate = sample_rate
#         self._bandwidth_min = bandwidth_min
#         self._bandwidth_max = bandwidth_max
#         self._db_min = db_min
#         self._db_max = db_max
#         self._db = db
#         self._num_bands = num_bands
#         self._min_freq = 100
#         self._max_freq = (self._sample_rate / 2) - 100
#         self._num_bands_min = num_bands_min
#         self._num_bands_max = num_bands_max

#     def compute_central_frequencies(self, freq_range, num_freqs):
#         # ... (rest of the method unchanged)

#     def __call__(self, data):
#         return self.process(data)[0]

#     def process(self, data):
#         spectra = librosa.stft(data, n_fft=2048, hop_length=512)
#         # orig_energy = np.sqrt(np.mean(np.abs(spectra)**2))
#         out_bandwidths = []  # Changed from out_qs to out_bandwidths
#         out_dbs = []
#         num_bands = random.randint(self._num_bands_min, self._num_bands_max)
#         central_freqs = self.compute_central_frequencies((self._min_freq, self._max_freq), num_bands)

#         for i in range(num_bands):
#             bandwidth = random.uniform(self._bandwidth_min, self._bandwidth_max)  # Random bandwidth value between bandwidth_min and bandwidth_max
#             if self._db is not None:
#                 db = self._db[i]
#             else:
#                 db = random.uniform(self._db_min, self._db_max)

#             freq = central_freqs[i]
#             low = int(np.round((freq - bandwidth/2) / (self._sample_rate / 2048)))
#             high = int(np.round((freq + bandwidth/2) / (self._sample_rate / 2048)))
#             gain = 10**(db / 20)
#             spectra[low:high] *= gain

#             out_bandwidths.append(bandwidth)  # Changed from out_qs to out_bandwidths
#             out_dbs.append(db)

#         data = librosa.istft(spectra, n_fft=2048, hop_length=512, length=len(data))
#         # processed_energy = np.sqrt(np.mean(data**2))
#         # data *= (orig_energy / (processed_energy + 1e-8))
#         return data, out_bandwidths, out_dbs  # Changed from out_qs to out_bandwidths


@jit(nopython=True)
def EQ_process_band(
    spectra, q_min, q_max, db_min, db_max, min_freq, max_freq, sample_rate, bandwidth_max
):
    q = np.random.uniform(q_min, q_max)
    db = np.random.uniform(db_min, db_max)
    freq = np.random.uniform(min_freq, max_freq)  # Random central frequency
    bandwidth = freq / q
    if bandwidth > bandwidth_max:
        bandwidth = bandwidth_max
    low = int(np.round((freq - bandwidth / 2) / (sample_rate / 2048)))
    high = int(np.round((freq + bandwidth / 2) / (sample_rate / 2048)))
    # Ensure indices are within bounds
    low = max(0, min(low, len(spectra) - 1))
    high = max(0, min(high, len(spectra)))
    gain = 10 ** (db / 20)
    spectra[low:high] *= gain
    return spectra, q, db


class EQPerturbFreq:
    def __init__(
        self,
        sample_rate,
        q_min=0.5,
        q_max=3,
        q=None,
        db_min=-10,
        db_max=0,
        db=None,
        num_bands_min=1,
        num_bands_max=5,
        num_bands=None,
        bandwidth_max=6000,
    ):
        self._sample_rate = sample_rate
        self._q_min = q_min
        self._q_max = q_max
        self._q = q
        self._db_min = db_min
        self._db_max = db_max
        self._db = db
        self._num_bands = num_bands
        self._min_freq = 100
        self._max_freq = (self._sample_rate / 2) - 100
        self._num_bands_min = num_bands_min
        self._num_bands_max = num_bands_max
        self._bandwidth_max = bandwidth_max

    def __call__(self, data):
        return self.process(data)[0]

    def process(self, data):
        spectra = librosa.stft(data, n_fft=2048, hop_length=512)
        out_qs = []
        out_dbs = []
        num_bands = np.random.randint(self._num_bands_min, self._num_bands_max + 1)

        for _ in range(num_bands):
            spectra, q, db = EQ_process_band(
                spectra,
                self._q_min,
                self._q_max,
                self._db_min,
                self._db_max,
                self._min_freq,
                self._max_freq,
                self._sample_rate,
                self._bandwidth_max,
            )
            out_qs.append(q)
            out_dbs.append(db)

        data = librosa.istft(spectra, n_fft=2048, hop_length=512, length=len(data))
        return data, out_qs, out_dbs


# class EQPerturb:
#     def __init__(self, sample_rate, q_min=0.5, q_max=1.5, q=None, db_min=-10, db_max=0, db=None, num_bands_min=1, num_bands_max=10, num_bands=None):
#         self._sample_rate = sample_rate
#         self._q_min = q_min
#         self._q_max = q_max
#         self._q = q
#         self._db_min = db_min
#         self._db_max = db_max
#         self._db = db
#         self._num_bands = num_bands
#         self._min_freq = 100
#         self._max_freq = (self._sample_rate / 2) - 100
#         self._num_bands_min = num_bands_min
#         self._num_bands_max = num_bands_max

#     def compute_central_frequencies(self, freq_range, num_freqs):
#         # Check if the input parameters are valid
#         if not isinstance(freq_range, tuple) or len(freq_range) != 2:
#             raise ValueError("freq_range must be a tuple of two numbers")
#         if not isinstance(num_freqs, int) or num_freqs <= 0:
#             raise ValueError("num_freqs must be a positive integer")
#         # Convert the frequency range to logarithmic scale
#         log_range = np.log10(freq_range)
#         # Compute the logarithmic intervals
#         log_intervals = np.linspace(log_range[0], log_range[1], num_freqs + 1)
#         # Compute the geometric mean of each interval as the central frequency
#         central_freqs = 10 ** ((log_intervals[:-1] + log_intervals[1:]) / 2)
#         # Return the central frequencies as a numpy array
#         return central_freqs

#     def __call__(self, data):
#         return self.process(data)[0]

#     def process(self, data):
#         spectra = librosa.stft(data, n_fft=2048, hop_length=512)
#         # orig_energy = np.sqrt(np.mean(np.abs(spectra)**2))
#         out_qs = []
#         out_dbs = []
#         num_bands = random.randint(self._num_bands_min, self._num_bands_max)
#         central_freqs = self.compute_central_frequencies((self._min_freq, self._max_freq), num_bands)

#         for i in range(num_bands):
#             if self._q is not None:
#                 q = self._q[i]
#             else:
#                 q = random.uniform(self._q_min, self._q_max)
#             if self._db is not None:
#                 db = self._db[i]
#             else:
#                 db = random.uniform(self._db_min, self._db_max)

#             freq = central_freqs[i]
#             bandwidth = freq / q
#             low = int(np.round((freq - bandwidth/2) / (self._sample_rate / 2048)))
#             high = int(np.round((freq + bandwidth/2) / (self._sample_rate / 2048)))
#             gain = 10**(db / 20)
#             spectra[low:high] *= gain

#             out_qs.append(q)
#             out_dbs.append(db)

#         data = librosa.istft(spectra, n_fft=2048, hop_length=512, length=len(data))
#         # processed_energy = np.sqrt(np.mean(data**2))
#         # data *= (orig_energy / (processed_energy + 1e-8))
#         return data, out_qs, out_dbs

# ... rest of your code ...


class EQPerturbTime:
    # q = central frequency / bandwidth
    def __init__(
        self,
        sample_rate,
        q_min=0.5,
        q_max=3,
        q=None,
        db_min=-10,
        db_max=0,
        db=None,
        num_bands_min=1,
        num_bands_max=5,
        num_bands=None,
        bandwith_max=6000,
    ):
        self._sample_rate = sample_rate
        self._q_min = q_min
        self._q_max = q_max
        self._q = q
        self._db_min = db_min
        self._db_max = db_max
        self._db = db
        self._num_bands = num_bands
        self._min_freq = 100
        self._max_freq = (self._sample_rate / 2) - 100
        self._num_bands_min = num_bands_min
        self._num_bands_max = num_bands_max
        self._bandwith_max = bandwith_max

    def compute_central_frequencies(self, freq_range, num_freqs):
        # Check if the input parameters are valid
        if not isinstance(freq_range, tuple) or len(freq_range) != 2:
            raise ValueError("freq_range must be a tuple of two numbers")
        if not isinstance(num_freqs, int) or num_freqs <= 0:
            raise ValueError("num_freqs must be a positive integer")
        # Convert the frequency range to logarithmic scale
        log_range = np.log10(freq_range)
        # Compute the logarithmic intervals
        log_intervals = np.linspace(log_range[0], log_range[1], num_freqs + 1)
        # Compute the geometric mean of each interval as the central frequency
        central_freqs = 10 ** ((log_intervals[:-1] + log_intervals[1:]) / 2)
        # Return the central frequencies as a numpy array
        return central_freqs

    def __call__(self, data):
        return self.process(data)[0]

    def process(self, data):
        data = torch.from_numpy(data)
        # orig_energy = torch.sqrt(torch.mean(data ** 2, dim=0, keepdim=True))
        out_qs = []
        out_dbs = []
        num_bands = np.random.randint(self._num_bands_min, self._num_bands_max + 1)
        central_freqs = self.compute_central_frequencies(
            (self._min_freq, self._max_freq), num_bands
        )
        # apply_audio_effects = AudioEffectsChain()
        for i in range(num_bands):
            if self._q is not None:
                q = self._q[i]
            else:
                q = np.random.uniform(self._q_min, self._q_max)
            if self._db is not None:
                db = self._db[i]
            else:
                db = np.random.uniform(self._db_min, self._db_max)
            # apply_audio_effects = apply_audio_effects.equalizer(frequency=self._central_freqs[i], q=q, db=db)
            if central_freqs[i] / q > self._bandwith_max:
                q = central_freqs[i] / self._bandwith_max
            data = taf.equalizer_biquad(
                data, sample_rate=self._sample_rate, center_freq=central_freqs[i], gain=db, Q=q
            )
            # apply_audio_effects = AudioEffectsChain().equalizer(frequency=self._central_freqs[i], q=q, db=db)
            # data = data / np.abs(data).max() * 0.8
            # data = apply_audio_effects(src=data, sample_in=self._sample_rate)
            # processed_energy = torch.sqrt(torch.mean(data ** 2, dim=0, keepdim=True))
            # data = data * (orig_energy / (processed_energy + 1e-8))
            out_qs.append(q)
            out_dbs.append(db)
        data = data.numpy()
        return data, out_qs, out_dbs

    # def process_multi(self, noisy, clean, noise):
    #     noisy = torch.from_numpy(noisy)
    #     clean = torch.from_numpy(clean)
    #     noise = torch.from_numpy(noise)
    #     orig_energy = torch.sqrt(torch.mean(noisy ** 2, dim=0, keepdim=True))
    #     out_qs = []
    #     out_dbs = []
    #     # apply_audio_effects = AudioEffectsChain()
    #     for i in range(self._num_bands):
    #         if self._q is not None:
    #             q = self._q[i]
    #         else:
    #             q = random.uniform(self._q_min, self._q_max)
    #         if self._db is not None:
    #             db = self._db[i]
    #         else:
    #             db = random.uniform(self._db_min, self._db_max)
    #         # apply_audio_effects = apply_audio_effects.equalizer(frequency=self._central_freqs[i], q=q, db=db)
    #         noisy = taf.equalizer_biquad(noisy, sample_rate=self._sample_rate, center_freq=self._central_freqs[i], gain=db, Q=q)
    #         clean = taf.equalizer_biquad(clean, sample_rate=self._sample_rate, center_freq=self._central_freqs[i], gain=db, Q=q)
    #         noise = taf.equalizer_biquad(noise, sample_rate=self._sample_rate, center_freq=self._central_freqs[i], gain=db, Q=q)
    #         # apply_audio_effects = AudioEffectsChain().equalizer(frequency=self._central_freqs[i], q=q, db=db)
    #         # data = data / np.abs(data).max() * 0.8
    #         # data = apply_audio_effects(src=data, sample_in=self._sample_rate)
    #         processed_energy = torch.sqrt(torch.mean(noisy ** 2, dim=0, keepdim=True))
    #         noisy = noisy * (orig_energy / (processed_energy + 1e-8))
    #         clean = clean * (orig_energy / (processed_energy + 1e-8))
    #         noise = noise * (orig_energy / (processed_energy + 1e-8))
    #         out_qs.append(q)
    #         out_dbs.append(db)
    #     noisy = noisy.numpy()
    #     clean = clean.numpy()
    #     noise = noise.numpy()
    #     return noisy, clean, noise, out_qs, out_dbs

    # def process_noise(self, noisy, clean, noise):
    #     noise = torch.from_numpy(noise)
    #     orig_energy = torch.sqrt(torch.mean(noise ** 2, dim=0, keepdim=True))
    #     out_qs = []
    #     out_dbs = []
    #     # apply_audio_effects = AudioEffectsChain()
    #     for i in range(self._num_bands):
    #         if self._q is not None:
    #             q = self._q[i]
    #         else:
    #             q = random.uniform(self._q_min, self._q_max)
    #         if self._db is not None:
    #             db = self._db[i]
    #         else:
    #             db = random.uniform(self._db_min, self._db_max)
    #         # apply_audio_effects = apply_audio_effects.equalizer(frequency=self._central_freqs[i], q=q, db=db)

    #         noise = taf.equalizer_biquad(noise, sample_rate=self._sample_rate, center_freq=self._central_freqs[i], gain=db, Q=q)
    #         # apply_audio_effects = AudioEffectsChain().equalizer(frequency=self._central_freqs[i], q=q, db=db)
    #         # data = data / np.abs(data).max() * 0.8
    #         # data = apply_audio_effects(src=data, sample_in=self._sample_rate)
    #         processed_energy = torch.sqrt(torch.mean(noise ** 2, dim=0, keepdim=True))
    #         noise = noise * (orig_energy / (processed_energy + 1e-8))
    #         out_qs.append(q)
    #         out_dbs.append(db)
    #     noisy = clean + noise
    #     return noisy, clean, noise, out_qs, out_dbs


class EQPerturb:
    def __init__(self, sample_rate=48000, db_min=-12, db_max=12):
        self.sample_rate = sample_rate
        self.db_min = db_min
        self.db_max = db_max
        self.n_bands_min = 5
        self.n_bands_max = 20

    def generate_gain_values(self, n_bands):
        num_selected_bands = n_bands
        return np.random.uniform(self.db_min, self.db_max, num_selected_bands)

    def apply_gain(self, freqs, stft_data, gain_values, frequency_bands):
        for index, gain_db in enumerate(gain_values):
            lowcut, highcut = frequency_bands[index]
            gain_max = gain_db
            freqs_to_apply = np.where((freqs >= lowcut) & (freqs <= highcut))[0]
            # 平滑gain值的窗函数
            window = signal.windows.hamming(len(freqs_to_apply))

            # 为每个频率点计算增益
            for f_index, f in enumerate(freqs):
                if lowcut <= f <= highcut:
                    gain = gain_max * window[f_index - freqs_to_apply[0]]
                    stft_data[f_index] *= 10 ** (gain / 20)

        return stft_data

    def process(self, data):
        n_bands = np.random.randint(self.n_bands_min, self.n_bands_max + 1)

        n_bands_in_use = np.random.randint(1, (n_bands + 1) // 2)

        frequency_bands = np.geomspace(10, self.sample_rate / 2, n_bands + 1)

        frequency_bands = [frequency_bands[i : i + 2] for i in range(len(frequency_bands) - 1)]

        frequency_bands_in_use = random.sample(frequency_bands, n_bands_in_use)

        gain_values = self.generate_gain_values(n_bands_in_use)

        # STFT
        Zxx = librosa.stft(data, n_fft=2048, hop_length=512, win_length=2048, center=True)

        # Frequency values for each FFT bin
        freqs = librosa.fft_frequencies(sr=self.sample_rate, n_fft=2048)
        # Apply gain in frequency domain
        Zxx_modified = self.apply_gain(freqs, Zxx, gain_values, frequency_bands_in_use)
        # ISTFT
        processed_data = librosa.istft(
            Zxx_modified, hop_length=512, win_length=2048, center=True, length=len(data)
        )

        return processed_data

    def __call__(self, data):
        return self.process(data)


class EQMuchGainPerturb:
    def __init__(self, sample_rate=48000, db_min=-12, db_max=12, freq_min=1000, freq_max=16000):
        self.sample_rate = sample_rate
        self.db_min = db_min
        self.db_max = db_max
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.n_bands_min = 12
        self.n_bands_max = 25

    def generate_gain_values(self, n_bands):
        num_selected_bands = n_bands
        return np.random.uniform(self.db_min, self.db_max, num_selected_bands)

    def apply_gain(self, freqs, stft_data, gain_values, frequency_bands):
        for index, gain_db in enumerate(gain_values):
            lowcut, highcut = frequency_bands[index]
            gain_max = gain_db
            freqs_to_apply = np.where((freqs >= lowcut) & (freqs <= highcut))[0]
            # 平滑gain值的窗函数
            window = signal.windows.hamming(len(freqs_to_apply))

            # 为每个频率点计算增益
            for f_index, f in enumerate(freqs):
                if lowcut <= f <= highcut:
                    gain = gain_max * window[f_index - freqs_to_apply[0]]
                    stft_data[f_index] *= 10 ** (gain / 20)

        return stft_data

    def process(self, data):
        n_bands = np.random.randint(self.n_bands_min, self.n_bands_max + 1)

        frequency_bands = np.geomspace(10, self.sample_rate / 2, n_bands + 1)

        frequency_bands = [x for x in frequency_bands if self.freq_min <= x <= self.freq_max]

        frequency_bands = [frequency_bands[i : i + 2] for i in range(len(frequency_bands) - 1)]

        n_bands_in_use = np.random.randint(1, min(len(frequency_bands) // 2 + 1, 3))

        frequency_bands_in_use = random.sample(frequency_bands, n_bands_in_use)

        gain_values = self.generate_gain_values(n_bands_in_use)

        # STFT
        Zxx = librosa.stft(data, n_fft=2048, hop_length=512, win_length=2048, center=True)

        # Frequency values for each FFT bin
        freqs = librosa.fft_frequencies(sr=self.sample_rate, n_fft=2048)
        # Apply gain in frequency domain
        Zxx_modified = self.apply_gain(freqs, Zxx, gain_values, frequency_bands_in_use)
        # ISTFT
        processed_data = librosa.istft(
            Zxx_modified, hop_length=512, win_length=2048, center=True, length=len(data)
        )

        return processed_data

    def __call__(self, data):
        return self.process(data)


# # 使用
# eq_perturb = EQPerturb()
# processed_data = eq_perturb.process(your_audio_data)  # your_audio_data 应是一个NumPy数组


class BassBoostPerturb:
    def __init__(
        self,
        sample_rate,
        highpass_cutoff_min=500,
        highpass_cutoff_max=2000,
        attenuation_min_db=-20,
    ):
        self._sample_rate = sample_rate
        self._highpass_cutoff_min = highpass_cutoff_min
        self._highpass_cutoff_max = highpass_cutoff_max
        self._attenuation_min_db = attenuation_min_db

    def __call__(self, data):
        return self.process(data)

    def process(self, data):
        cutoff_freq = np.random.uniform(self._highpass_cutoff_min, self._highpass_cutoff_max)
        attenuation_factor = 10 ** (np.random.uniform(self._attenuation_min_db, 0) / 20)

        # 设计高通滤波器来提取高频成分
        b_high, a_high = signal.butter(
            N=4, Wn=cutoff_freq / (0.5 * self._sample_rate), btype="high"
        )

        # 应用高通滤波器并进行衰减调整
        high_freq_data = signal.filtfilt(b_high, a_high, data)
        adjusted_high_freq = high_freq_data * attenuation_factor
        adjusted_data = data - high_freq_data + adjusted_high_freq

        return adjusted_data


class DRCPerturb:
    def __init__(
        self,
        sample_rate,
        threshold_db_min=-30,
        threshold_db_max=0,
        threshold_db=None,
        ratio_min=1,
        ratio_max=20,
        ratio=None,
        attack_ms_min=0.5,
        attack_ms_max=2.0,
        attack_ms=None,
        release_ms_min=50,
        release_ms_max=200,
        release_ms=None,
    ):
        self._sample_rate = sample_rate
        self._threshold_db_min = threshold_db_min
        self._threshold_db_max = threshold_db_max
        self._threshold_db = threshold_db
        self._ratio_min = ratio_min
        self._ratio_max = ratio_max
        self._ratio = ratio
        self._attack_ms_min = attack_ms_min
        self._attack_ms_max = attack_ms_max
        self._attack_ms = attack_ms
        self._release_ms_min = release_ms_min
        self._release_ms_max = release_ms_max
        self._release_ms = release_ms

    def __call__(self, data):
        return self.process(data)

    def process(self, data):
        if self._threshold_db is not None:
            threshold_db = self._threshold_db
        else:
            threshold_db = np.random.uniform(self._threshold_db_min, self._threshold_db_max)
        if self._ratio is not None:
            ratio = self._ratio
        else:
            ratio = np.random.uniform(self._ratio_min, self._ratio_max)
        if self._attack_ms is not None:
            attack_ms = self._attack_ms
        else:
            attack_ms = np.random.uniform(self._attack_ms_min, self._attack_ms_max)
        if self._release_ms is not None:
            release_ms = self._release_ms
        else:
            release_ms = np.random.uniform(self._release_ms_min, self._release_ms_max)
        inst = Compressor(
            threshold_db=threshold_db, ratio=ratio, attack_ms=attack_ms, release_ms=release_ms
        )
        data = inst(data, self._sample_rate)
        return data


class SpeakerDistortionPerturbSox:
    def __init__(
        self,
        sample_rate,
        gain_db_min=10,
        gain_db_max=30,
        gain_db=None,
        slope_min=10,
        slope_max=30,
        slope=None,
    ):
        self._sample_rate = sample_rate
        self._gain_db_min = gain_db_min
        self._gain_db_max = gain_db_max
        self._gain_db = gain_db
        self._slope_min = slope_min
        self._slope_max = slope_max
        self._slope = slope

    def __call__(self, data):
        return self.process(data)

    def process(self, data):
        if self._gain_db is not None:
            gain_db = self._gain_db
        else:
            gain_db = np.random.uniform(self._gain_db_min, self._gain_db_max)
        if self._slope is not None:
            slope = self._slope
        else:
            slope = np.random.uniform(self._slope_min, self._slope_max)
        # apply_audio_effects = \
        #     (AudioEffectsChain()
        #     .overdrive(gain=gain_db, colour=slope))
        # orig_energy = np.sqrt(np.mean(data ** 2, axis=0, keepdims=True))
        # data = apply_audio_effects(src=data, sample_in=self._sample_rate)
        # processed_energy = np.sqrt(np.mean(data ** 2, axis=0, keepdims=True))
        # data = data * (orig_energy / processed_energy)
        data = torch.from_numpy(data)
        orig_energy = torch.sqrt(torch.mean(data**2, dim=0, keepdim=True))
        data = taf.overdrive(data, gain=gain_db, colour=slope)
        processed_energy = torch.sqrt(torch.mean(data**2, dim=0, keepdim=True))
        data = data * (orig_energy / (processed_energy + 1e-8))
        data = data.numpy()
        return data


class SpeakerDistortionPerturbPedal:
    def __init__(self, sample_rate, drive_db_min=10, drive_db_max=30, drive_db=None):
        self._sample_rate = sample_rate
        self._drive_db_min = drive_db_min
        self._drive_db_max = drive_db_max
        self._drive_db = drive_db

    def __call__(self, data):
        return self.process(data)

    def process(self, data):
        if self._drive_db is not None:
            drive_db = self._drive_db
        else:
            drive_db = np.random.uniform(self._drive_db_min, self._drive_db_max)
        inst = Distortion(drive_db=drive_db)
        orig_energy = np.sqrt(np.mean(data**2, axis=0, keepdims=True))
        data = inst(data, self._sample_rate)
        processed_energy = np.sqrt(np.mean(data**2, axis=0, keepdims=True))
        data = data * (orig_energy / (processed_energy + 1e-8))
        return data


class SpeakerDistortionPerturbClipPedal:
    def __init__(self, sample_rate, threshold_db_min=-20, threshold_db_max=-1, threshold_db=None):
        self._sample_rate = sample_rate
        self._threshold_db_min = threshold_db_min
        self._threshold_db_max = threshold_db_max
        self._threshold_db = threshold_db

    def __call__(self, data):
        return self.process(data)

    def process(self, data):
        if self._threshold_db is not None:
            threshold_db = self._threshold_db
        else:
            threshold_db = np.random.uniform(self._threshold_db_min, self._threshold_db_max)
        inst = Clipping(threshold_db=threshold_db)
        data = inst(data, self._sample_rate)
        return data


class SpeakerDistortionPerturbHardClip:
    def __init__(self, sample_rate, threshold_db_min=-20, threshold_db_max=-1, threshold_db=None):
        self._sample_rate = sample_rate
        self._threshold_db_min = threshold_db_min
        self._threshold_db_max = threshold_db_max
        self._threshold_db = threshold_db

    def __call__(self, data):
        return self.process(data)

    def process(self, data):
        if self._threshold_db is not None:
            threshold_db = self._threshold_db
        else:
            threshold_db = np.random.uniform(self._threshold_db_min, self._threshold_db_max)
        # convert threshold_db to threshold
        threshold = 10 ** (threshold_db / 20)
        # hard clip
        data = np.clip(data, -threshold, threshold)
        return data


class SpeakerDistortionPerturbHardClipOnRate:
    def __init__(self, sample_rate, clip_rate_min=0.01, clip_rate_max=0.3, clip_rate=None):
        self._sample_rate = sample_rate
        self._clip_rate_min = clip_rate_min
        self._clip_rate_max = clip_rate_max
        self._clip_rate = clip_rate

    def __call__(self, data):
        return self.process(data)

    def process(self, data):
        if self._clip_rate is not None:
            clip_rate = self._clip_rate
        else:
            clip_rate = np.random.uniform(self._clip_rate_min, self._clip_rate_max)
        hist, bin_edges = np.histogram(np.abs(data), bins=1000)
        clip_threshold = bin_edges[:-1][np.cumsum(hist) > (1 - clip_rate) * len(data)][0]
        data = np.clip(data, -clip_threshold, clip_threshold)
        return data


class SpeakerDistortionPerturbSoftClip:
    def __init__(self, sample_rate, slope_min=1, slope_max=5, slope=None):
        self._sample_rate = sample_rate
        self._slope_min = slope_min
        self._slope_max = slope_max
        self._slope = slope

    def __call__(self, data):
        return self.process(data)

    def process(self, data):
        if self._slope is not None:
            slope = self._slope
        else:
            slope = np.random.uniform(self._slope_min, self._slope_max)
        # soft clip satuation function
        x_max = data.max()
        data = (
            x_max * data / (np.abs(x_max) ** slope + np.abs(data) ** slope + 1e-5) ** (1 / slope)
        )

        return data


class SpeakerDistortionPerturbSigmoid1:
    def __init__(
        self,
        sample_rate,
        slope_min=1,
        slope_max=5,
        slope=None,
        shape_min=1,
        shape_max=5,
        shape=None,
    ):
        self._sample_rate = sample_rate
        self._slope_min = slope_min
        self._slope_max = slope_max
        self._slope = slope
        self._shape_min = shape_min
        self._shape_max = shape_max
        self._shape = shape

    def __call__(self, data):
        return self.process(data)

    def process(self, data):
        if self._slope is not None:
            slope = self._slope
        else:
            slope = np.random.uniform(self._slope_min, self._slope_max)
        if self._shape is not None:
            shape = self._shape
        else:
            shape = np.random.uniform(self._shape_min, self._shape_max)
        # sigmoid function
        orig_energy = np.sqrt(np.mean(data**2, axis=0, keepdims=True))
        data = (2 / (1 + np.exp(-slope * data)) - 1) * shape
        processed_energy = np.sqrt(np.mean(data**2, axis=0, keepdims=True))
        data = data * (orig_energy / (processed_energy + 1e-8))
        return data


class SpeakerDistortionPerturbSigmoid2:
    def __init__(
        self,
        sample_rate,
        threshold_db_min=-10,
        threshold_db_max=-1,
        threshold_db=None,
        gain_min=1,
        gain_max=4,
        gain=None,
    ):
        self._sample_rate = sample_rate
        self._threshold_db_min = threshold_db_min
        self._threshold_db_max = threshold_db_max
        self._threshold_db = threshold_db
        self._gain_min = gain_min
        self._gain_max = gain_max
        self._gain = gain

    def __call__(self, data):
        return self.process(data)

    def process(self, data):
        if self._threshold_db is not None:
            threshold_db = self._threshold_db
        else:
            threshold_db = np.random.uniform(self._threshold_db_min, self._threshold_db_max)
        if self._gain is not None:
            gain = self._gain
        else:
            gain = np.random.uniform(self._gain_min, self._gain_max)

        orig_energy = np.sqrt(np.mean(data**2, axis=0, keepdims=True))
        # convert threshold_db to threshold
        threshold = 10 ** (threshold_db / 20)
        # sigmoid function
        x_clip = np.clip(data, -threshold, threshold)
        b = 1.5 * x_clip - 0.3 * x_clip**2
        a = np.ones_like(b) * 0.5
        a[b > 0] = 4
        data = gain * (2 / (1 + np.exp(-a * b)) - 1)
        processed_energy = np.sqrt(np.mean(data**2, axis=0, keepdims=True))
        data = data * (orig_energy / (processed_energy + 1e-8))
        return data


class LoudnessPerturb:
    def __init__(self, sample_rate, min_factor=0.1, max_factor=10.0, max_n_intervals=5):
        assert 0.0 < min_factor < 1.0, "min_factor >= 1. or <=0."
        assert max_factor > 1.0, "max_factor <= 1."
        assert max_n_intervals > 0, "max_n_intervals <= 0"
        self._sample_rate = sample_rate
        self._min_factor = min_factor
        self._max_factor = max_factor
        self._max_n_intervals = max_n_intervals

    def __call__(self, data):
        assert data.shape[0] > 0, "waveform is empty"
        data = data.copy()
        n_intervals = np.random.randint(1, self._max_n_intervals + 1)

        len_interval = len(data) // n_intervals
        for i in range(n_intervals):
            if np.random.uniform() < 0.5:
                factor = np.random.uniform(self._min_factor, 1.0)  # reduce loudness
            else:
                factor = np.random.uniform(1.0, self._max_factor)  # increase loudness
            data[i * len_interval : (i + 1) * len_interval] = (
                factor * data[i * len_interval : (i + 1) * len_interval]
            )
        return data


class LowPassPerturb:
    def __init__(
        self, sample_rate, min_cutoff_freq=1000, max_cutoff_freq=24000, min_order=4, max_order=20
    ):
        assert min_cutoff_freq >= 0, "min_cutoff_freq < 0"
        assert max_cutoff_freq > 0, "max_cutoff_freq <= 0"
        self._sample_rate = sample_rate
        self._min_cutoff_freq = min_cutoff_freq
        self._max_cutoff_freq = max_cutoff_freq
        self._min_order = min_order
        self._max_order = max_order

    def low_pass_stft(self, data, cutoff_freq):
        freqs = librosa.fft_frequencies(sr=self._sample_rate, n_fft=2048)
        mask = freqs > cutoff_freq
        data[mask] = 0
        return data

    def __call__(self, data):
        assert data.shape[0] > 0, "waveform is empty"
        data = data.copy()
        cutoff_freq = np.random.uniform(self._min_cutoff_freq, self._max_cutoff_freq)
        if np.random.random() < 0.3:
            orig_len = len(data)
            data = librosa.stft(data, n_fft=2048, hop_length=512)
            data = self.low_pass_stft(data, cutoff_freq)
            data = librosa.istft(data, n_fft=2048, hop_length=512, length=orig_len)
        else:
            order = np.random.randint(self._min_order, self._max_order + 1)
            sos = signal.butter(order, cutoff_freq, "lp", fs=self._sample_rate, output="sos")
            data = signal.sosfilt(sos, data)
        return data


class BandRejectPerturb:
    def __init__(
        self,
        sample_rate,
        min_center_freq=1000,
        max_center_freq=8000,
        min_q=5,
        max_q=10,
        min_freq_bandwidth=100,
        max_freq_bandwidth=2000,
        use_stft=False,
        max_n=2,
    ):
        self._sample_rate = sample_rate
        self._min_center_freq = min_center_freq
        self._max_center_freq = max_center_freq
        self._min_q = min_q
        self._max_q = max_q
        self._min_freq_bandwidth = min_freq_bandwidth
        self._max_freq_bandwidth = max_freq_bandwidth
        self._use_stft = use_stft
        self._max_n = max_n

    def band_reject_stft(self, data, center_freq, freq_bandwidth):
        freq_bandwidth = min(freq_bandwidth, center_freq / 2)
        freqs = librosa.fft_frequencies(sr=self._sample_rate, n_fft=2048)
        min_freq = center_freq - freq_bandwidth / 2
        max_freq = center_freq + freq_bandwidth / 2
        mask = np.logical_and(freqs >= min_freq, freqs <= max_freq)
        data[mask] = 0
        return data

    def __call__(self, data):
        curent_n = np.random.randint(1, self._max_n + 1)
        if self._use_stft:
            orig_len = len(data)
            data = librosa.stft(data, n_fft=2048, hop_length=512)
            for i in range(curent_n):
                center_freq = np.random.uniform(self._min_center_freq, self._max_center_freq)
                q = np.random.uniform(self._min_q, self._max_q)
                freq_bandwidth = np.random.uniform(
                    self._min_freq_bandwidth, self._max_freq_bandwidth
                )
                data = self.band_reject_stft(data, center_freq, freq_bandwidth)
            data = librosa.istft(data, n_fft=2048, hop_length=512, length=orig_len)
        else:
            for i in range(curent_n):
                center_freq = np.random.uniform(self._min_center_freq, self._max_center_freq)
                q = np.random.uniform(self._min_q, self._max_q)
                freq_bandwidth = np.random.uniform(
                    self._min_freq_bandwidth, self._max_freq_bandwidth
                )
                b, a = signal.iirnotch(center_freq, q, fs=self._sample_rate)
                data = signal.lfilter(b, a, data)
        return data


# class SpectralLeakagePerturb:
#     def __init__(self, sample_rate, window_lengths=[1024, 2048, 4096], max_time_shift=10):
#         self.sample_rate = sample_rate
#         self.window_lengths = window_lengths
#         self.max_time_shift = max_time_shift

#     def __call__(self, data):
#         time_shift = np.random.randint(-self.max_time_shift, self.max_time_shift)
#         self.window_length = random.choice(self.window_lengths)
#         window = signal.hann(self.window_length)
#         framed = signal.stft(data, nperseg=self.window_length, window=window)[2]

#         # apply phase shift
#         phases = np.angle(framed)
#         phases = np.roll(phases, time_shift)

#         # reconstruct with modified phase
#         framed = np.abs(framed) * np.exp(1j * phases)
#         return signal.istft(framed, window=window)[1]


class SpectralLeakagePerturb:
    def __init__(self, sample_rate, window_lengths=[1024, 2048, 4096], max_time_shift=10):
        self.sample_rate = sample_rate
        self.window_lengths = window_lengths
        self.max_time_shift = max_time_shift

    def __call__(self, data):
        time_shift = np.random.randint(-self.max_time_shift, self.max_time_shift)
        self.window_length = random.choice(self.window_lengths)
        window = "hann"

        # Use librosa's stft
        framed = librosa.stft(
            data,
            n_fft=self.window_length,
            win_length=self.window_length,
            hop_length=self.window_length // 4,
            window=window,
            center=True,
        )

        # apply phase shift
        phases = np.angle(framed)
        phases = np.roll(phases, time_shift, axis=-1)  # make sure the shift is along the time axis

        # reconstruct with modified phase
        framed = np.abs(framed) * np.exp(1j * phases)

        # Use librosa's istft
        return librosa.istft(
            framed,
            n_fft=self.window_length,
            win_length=self.window_length,
            hop_length=self.window_length // 4,
            window=window,
            center=True,
            length=len(data),
        )


class DCOffsetPerturb:
    def __init__(self, sample_rate, min_offset=0.1, max_offset=0.5):
        self.sample_rate = sample_rate
        self.min_offset = min_offset
        self.max_offset = max_offset

    def __call__(self, data):
        offset = np.random.uniform(self.min_offset, self.max_offset)
        return data + offset


class WhiteNoisePerturb:
    def __init__(self, sample_rate, snr_min, snr_max):
        self.sample_rate = sample_rate
        self.snr_min = snr_min
        self.snr_max = snr_max

    def __call__(self, data):
        snr_db = np.random.uniform(self.snr_min, self.snr_max)
        snr = 10 ** (snr_db / 20)
        noise_level = np.sqrt(np.mean(data**2)) / snr
        noise = np.random.randn(*data.shape)
        noisy = data + noise_level * noise
        return noisy


class ColoredNoisePerturb:
    def __init__(
        self,
        sample_rate=44100,
        snr_min=10,
        snr_max=30,
        color_types=["white", "pink", "brown", "equalized"],
    ):
        self.sample_rate = sample_rate
        self.snr_min = snr_min
        self.snr_max = snr_max
        self.color_types = color_types

        self.num_samples = None
        self.noise = None
        self.filter_b, self.filter_a = None, None

    def generate_white_noise(self, num_samples):
        return np.random.normal(0, 1, num_samples)

    def generate_pink_noise(self, num_samples):
        pink_filter_order = np.random.randint(1, 5)
        W = np.random.uniform(0.01, 0.9)
        b, a = signal.butter(pink_filter_order, W, "low", analog=False)
        pink_noise = np.random.normal(0, 1, num_samples)
        pink_noise = signal.lfilter(b, a, pink_noise)
        pink_noise /= np.max(np.abs(pink_noise))  # Normalize to [-1, 1]
        return pink_noise

    def generate_brown_noise(self, num_samples):
        brown_noise = np.cumsum(np.random.normal(0, 1, num_samples))
        brown_noise -= np.mean(brown_noise)
        brown_noise /= np.max(np.abs(brown_noise))  # Normalize to [-1, 1]
        return brown_noise

    def generate_equalized_noise(self, num_samples):
        # randomly equalized white noise
        white_noise = np.random.normal(0, 1, num_samples)
        # random euqalizer
        num_bands = np.random.randint(1, 11)
        central_freqs = np.geomspace(100, self.sample_rate / 2 - 8000, num_bands)
        q = 1
        db = np.random.uniform(-20, 20, num_bands)
        for i in range(num_bands):
            b, a = signal.iirpeak(central_freqs[i], q, fs=self.sample_rate)
            adjust_noise = signal.lfilter(b, a, white_noise)
            res_noise = white_noise - adjust_noise
            adjust_noise *= 10 ** (db[i] / 20)
            white_noise = res_noise + adjust_noise
        white_noise /= np.max(np.abs(white_noise))  # Normalize to [-1, 1]
        return white_noise

    def apply_snr(self, data):
        snr = np.random.uniform(self.snr_min, self.snr_max)
        signal_power = np.mean(data**2)
        noise_power = signal_power / (10 ** (snr / 10))
        return noise_power

    def __call__(self, data):
        data_len = len(data)
        self.num_samples = data_len
        self.color = random.choice(self.color_types)

        if self.color == "white":
            noise = self.generate_white_noise(self.num_samples)
        elif self.color == "pink":
            noise = self.generate_pink_noise(self.num_samples)
        elif self.color == "brown":
            noise = self.generate_brown_noise(self.num_samples)
        elif self.color == "equalized":
            noise = self.generate_equalized_noise(self.num_samples)
        else:
            raise ValueError("Unsupported noise color. Supported colors: 'white', 'pink', 'brown'")

        noise_power = self.apply_snr(data)
        scaled_noise = np.sqrt(noise_power) * noise

        noisy_data = data + scaled_noise
        return noisy_data


class OPUSCodecsPerturb:
    def __init__(self, sample_rate, compress_rate_min=2, compress_rate_max=32):
        self._sample_rate = sample_rate
        self._compress_rate_min = compress_rate_min
        self._compress_rate_max = compress_rate_max
        self._applications = ["voip", "audio", "restricted_lowdelay"]
        self._vbrs = [0, 1]
        self._complexities = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        self._packet_loss_percs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self._bandwidths = [
            opuslib.constants.BANDWIDTH_FULLBAND,
            opuslib.constants.BANDWIDTH_SUPERWIDEBAND,
            opuslib.constants.BANDWIDTH_WIDEBAND,
        ]
        self.name = "OPUS"
        self.delay_min_ms = 0
        self.delay_max_ms = 100

    # def find_alignment_offset_wav(self, original, processed):
    #     # Compute cross-correlation
    #     correlation = np.correlate(processed, original, "full")
    #     # Find the offset where the maximum correlation occurs
    #     return np.argmax(correlation) - len(original) + 1

    # def find_alignment_offset(self, original, processed):
    #     # Zero-pad the signals for FFT
    #     len_total = len(original) + len(processed) - 1
    #     next_pow2 = 2**np.ceil(np.log2(len_total))
    #     original_padded = np.pad(original, (0, int(next_pow2) - len(original)))
    #     processed_padded = np.pad(processed, (0, int(next_pow2) - len(processed)))

    #     # Compute cross-correlation in the frequency domain
    #     original_fft = np.fft.fft(original_padded)
    #     processed_fft = np.fft.fft(processed_padded)
    #     correlation = np.fft.ifft(original_fft * np.conj(processed_fft)).real

    #     # Find the offset where the maximum correlation occurs
    #     return np.argmax(correlation) - len(original) + 1

    # 使用向量内积在给定的delay_min_ms和delay_max_ms之间找到最佳的延迟
    def find_alignment_offset(self, original, processed):
        delay_min = int(self._sample_rate * self.delay_min_ms / 1000)
        delay_max = int(self._sample_rate * self.delay_max_ms / 1000)
        # 使用时域卷积计算
        for delay in range(delay_min, delay_max):
            p_1 = processed[delay:]
            o_1 = original[: len(p_1)]
            if len(p_1) > len(o_1):
                p_1 = p_1[: len(o_1)]
            corr = np.inner(o_1, p_1)
            if delay == delay_min:
                max_corr = corr
                max_delay = delay
            else:
                if corr > max_corr:
                    max_corr = corr
                    max_delay = delay
        return max_delay

    def __call__(self, data):
        compress_rate = np.random.uniform(self._compress_rate_min, self._compress_rate_max)
        application = random.choice(self._applications)
        bitrate = int(self._sample_rate * 16 / compress_rate)
        vbr = random.choice(self._vbrs)
        complexity = random.choice(self._complexities)
        packet_loss_perc = random.choice(self._packet_loss_percs)
        bandwidth = random.choice(self._bandwidths)

        encoder = opuslib.Encoder(self._sample_rate, channels=1, application=application)
        encoder._set_bandwidth(bandwidth)
        # encoder._set_lsb_depth(16)
        encoder._set_complexity(complexity)
        encoder._set_bitrate(bitrate)
        encoder._set_vbr(vbr)
        encoder._set_packet_loss_perc(packet_loss_perc)
        decoder = opuslib.Decoder(self._sample_rate, channels=1)

        frame_size = self._sample_rate // 100
        original_len = len(data)
        if len(data) % frame_size != 0:
            data = np.pad(data, (0, frame_size - len(data) % frame_size), "constant")
        output_data = np.zeros_like(data)
        for i in range(0, len(data), frame_size):
            start = i
            end = min(i + frame_size, len(data))
            frame = (data[start:end] * 32768).astype(np.int16)
            frame = frame.tobytes()
            encoded_frame = encoder.encode(frame, end - start)
            decoded_frame = decoder.decode(encoded_frame, end - start)
            decoded_frame = np.frombuffer(decoded_frame, dtype=np.int16)
            output_data[start:end] = decoded_frame.astype(np.float32) / 32768.0

        # Find alignment offset
        if self._sample_rate == 48000:
            if application == "voip" or application == "audio":
                offset = 310
            if application == "restricted_lowdelay":
                offset = 120
        elif self._sample_rate == 24000:
            if application == "voip" or application == "audio":
                offset = 155
            if application == "restricted_lowdelay":
                offset = 60
        elif self._sample_rate == 16000:
            if application == "voip" or application == "audio":
                offset = 104
            if application == "restricted_lowdelay":
                offset = 40
        # offset = self.find_alignment_offset(data, output_data)
        # print(f"OPUS parameters: compress_rate={compress_rate}, application={application}, bitrate={bitrate}, vbr={vbr}, complexity={complexity}, packet_loss_perc={packet_loss_perc}, bandwidth={bandwidth}")
        # print(f"OPUS offset: {offset}")

        # # Align the audio using the offset
        # if offset > 0:
        #     aligned_audio = np.concatenate((np.zeros(offset), output_data))
        # else:
        #     aligned_audio = output_data[-offset:]
        aligned_audio = output_data[offset:]

        # Ensure the output audio has the same length as the input
        aligned_audio = aligned_audio[: len(data)]

        if aligned_audio.shape[0] < original_len:
            aligned_audio = np.pad(
                aligned_audio, (0, original_len - aligned_audio.shape[0]), "constant"
            )

        return aligned_audio


class GSMcodecsPerturb:
    def __init__(self, sample_rate):
        self._sample_rate = sample_rate
        self._qualitys = [pedalboard.Resample.Quality.WindowedSinc]
        self.name = "GSM"

    def __call__(self, data):
        quality = random.choice(self._qualitys)
        inst = GSMFullRateCompressor(quality=quality)
        output_data = inst(data, self._sample_rate)
        return output_data


class MP3CompressorPerturb:
    def __init__(self, sample_rate, vbr_min=1.0, vbr_max=9.5):
        self._sample_rate = sample_rate
        self.vbr_min = vbr_min
        self.vbr_max = vbr_max
        self.name = "MP3"

    def __call__(self, data):
        vbr = np.random.uniform(self.vbr_min, self.vbr_max)
        inst = MP3Compressor(vbr_quality=vbr)
        output_data = inst(data, self._sample_rate)
        return output_data


class BitCrushPerturb:
    def __init__(self, sample_rate, bit_min=4, bit_max=32):
        self._sample_rate = sample_rate
        self.bit_min = bit_min
        self.bit_max = bit_max

    def __call__(self, data):
        bit = np.random.randint(self.bit_min, self.bit_max + 1)
        inst = Bitcrush(bit_depth=bit)
        output_data = inst(data, self._sample_rate)
        return output_data


class PacketLossPerturb:
    def __init__(
        self,
        sample_rate,
        loss_rate_min=0,
        loss_rate_max=0.3,
        frame_time_min=0.008,
        frame_time_max=0.05,
        decay_rate_min=0,
        decay_rate_max=0.2,
        hard_loss_prob=1.0,
        loss_on_vad=False,
    ):
        self.sample_rate = sample_rate
        self.loss_rate_min = loss_rate_min
        self.loss_rate_max = loss_rate_max
        self.frame_time_min = frame_time_min
        self.frame_time_max = frame_time_max
        self.decay_rate_min = decay_rate_min
        self.decay_rate_max = decay_rate_max
        self.hard_loss_prob = hard_loss_prob
        self.loss_on_vad = loss_on_vad

    @staticmethod
    def webrtcvad_proc(data, sample_rate, webrtc_vad_mode):
        res = np.zeros_like(data)

        if sample_rate != 16000:
            # downsample
            data = librosa.resample(data, orig_sr=sample_rate, target_sr=16000)

        scale = 30000.0 / np.max(np.abs(data))
        data = data * scale
        data = data.astype(np.int16)
        vad = webrtcvad.Vad()
        vad.set_mode(webrtc_vad_mode)
        frame_len_samp = 160
        nf = int(data.shape[0] // frame_len_samp)
        frame_len_samp_in_res = int(sample_rate * frame_len_samp / 16000)
        for i in range(nf):
            vad_out = vad.is_speech(
                data[i * frame_len_samp : (i + 1) * frame_len_samp].tobytes(), 16000
            )
            res[i * frame_len_samp_in_res : (i + 1) * frame_len_samp_in_res] = vad_out
        return res

    def __call__(self, data):
        loss_rate = np.random.uniform(self.loss_rate_min, self.loss_rate_max)
        frame_time = np.random.uniform(self.frame_time_min, self.frame_time_max)
        frame_size = int(self.sample_rate * frame_time)
        if self.loss_on_vad:
            vad_res = self.webrtcvad_proc(data, self.sample_rate, 1)
            vad_res_frames = [
                vad_res[i : i + frame_size] for i in range(0, len(vad_res), frame_size)
            ]
        # Simulate packet loss by randomly dropping frames
        frames = [data[i : i + frame_size] for i in range(0, len(data), frame_size)]
        perturbed_frames = []

        for i, frame in enumerate(frames):
            if np.random.random() < loss_rate:
                new_p = np.random.random()
                if new_p < self.hard_loss_prob:
                    frame = np.zeros_like(frame)
                    perturbed_frames.append(frame)
                else:
                    decay_rate = np.random.uniform(self.decay_rate_min, self.decay_rate_max)
                    frame = frame * decay_rate
                    perturbed_frames.append(frame)
            else:
                perturbed_frames.append(frame)

        perturbed_data = np.concatenate(perturbed_frames)
        return perturbed_data


class AACConversionPerturb:
    def __init__(self, sample_rate=48000, compress_rate_min=2, compress_rate_max=32):
        self.sample_rate = sample_rate
        self._compress_rate_min = compress_rate_min
        self._compress_rate_max = compress_rate_max
        self.delay_min_ms = 5
        self.delay_max_ms = 50
        self.name = "AAC"
        if self.sample_rate == 48000 or self.sample_rate == 24000:
            self.dalay = 1024

    # def find_alignment_offset_wav(self, original, processed):
    #     # Compute cross-correlation
    #     correlation = np.correlate(processed, original, "full")
    #     # Find the offset where the maximum correlation occurs
    #     return np.argmax(correlation) - len(original) + 1

    # 使用向量内积在给定的delay_min_ms和delay_max_ms之间找到最佳的延迟
    def find_alignment_offset(self, original, processed):
        delay_min = int(self.sample_rate * self.delay_min_ms / 1000)
        delay_max = int(self.sample_rate * self.delay_max_ms / 1000)
        # 使用时域卷积计算
        for delay in range(delay_min, delay_max):
            p_1 = processed[delay:]
            o_1 = original[: len(p_1)]
            if len(p_1) > len(o_1):
                p_1 = p_1[: len(o_1)]
            corr = np.inner(o_1, p_1)
            if delay == delay_min:
                max_corr = corr
                max_delay = delay
            else:
                if corr > max_corr:
                    max_corr = corr
                    max_delay = delay
        return max_delay

    # def find_alignment_offset(self, original, processed):
    #     # Zero-pad the signals for FFT
    #     len_total = len(original) + len(processed) - 1
    #     # next_pow2 = 2**np.ceil(np.log2(len_total))
    #     original_padded = np.pad(original, (0, len_total - len(original)))
    #     processed_padded = np.pad(processed, (0, len_total - len(processed)))

    #     # Compute cross-correlation in the frequency domain
    #     original_fft = np.fft.fft(original_padded)
    #     processed_fft = np.fft.fft(processed_padded)
    #     correlation = np.fft.ifft(original_fft * np.conj(processed_fft)).real

    #     # Find the offset where the maximum correlation occurs
    #     return np.argmax(correlation) - len(original) + 1

    # def find_alignment_offset(self, original, processed):

    #     # Zero-pad the shorter signal
    #     len_orig = len(original)
    #     len_proc = len(processed)
    #     if len_orig < len_proc:
    #         original = np.pad(original, (0, len_proc - len_orig))
    #     else:
    #         processed = np.pad(processed, (0, len_orig - len_proc))

    #     # DFT
    #     original_dft = np.fft.fft(original)
    #     processed_dft = np.fft.fft(processed)

    #     # Normalize and multiply
    #     dft_prod = original_dft * np.conj(processed_dft) / (np.abs(original_dft) * np.abs(processed_dft))

    #     # IFFT
    #     corr = np.fft.ifft(dft_prod).real

    #     # Quadratic interpolation to find peak
    #     idx = np.argmax(corr)

    #     # 此处画出original，processed , original_dft, processed_dft 和 corr 的图像，观察是否有问题
    #     import matplotlib.pyplot as plt
    #     plt.subplot(5,1,1)
    #     plt.plot(original)
    #     plt.subplot(5,1,2)
    #     plt.plot(processed)
    #     plt.subplot(5,1,3)
    #     plt.plot(corr)
    #     plt.subplot(5,1,4)
    #     plt.plot(np.abs(original_dft))
    #     plt.subplot(5,1,5)
    #     plt.plot(np.abs(processed_dft))
    #     plt.show()

    #     if idx > 0 and idx < len(corr)-1:
    #         dx = (corr[idx+1] - corr[idx-1]) / (2 * (2*corr[idx] - corr[idx+1] - corr[idx-1]))
    #         peak = idx + dx
    #     else:
    #         peak = idx

    #     # Compute offset
    #     return int(peak - len_orig)

    def __call__(self, audio_data):
        compress_rate = np.random.uniform(self._compress_rate_min, self._compress_rate_max)
        bitrate = self.sample_rate * 16 / compress_rate
        bitrate = int(bitrate / 1000)

        audio_data_bytes = (audio_data * 32768).astype(np.int16).tobytes()

        # # Encoding
        # cmd_encode = (
        #     ffmpeg.input("pipe:0", format="s16le", ac=1, ar=self.sample_rate)
        #     .output("pipe:1", format="adts", ac=2, b=f"{bitrate}k")
        #     .global_args("-loglevel", "panic")
        #     .compile()
        # )
        # try:
        #     aac_audio_data, _ = subprocess.Popen(
        #         cmd_encode,
        #         stdin=subprocess.PIPE,
        #         stdout=subprocess.PIPE,
        #         stderr=subprocess.PIPE,
        #         bufsize=-1,
        #     ).communicate(input=audio_data_bytes)
        # except Exception as e:
        #     raise Exception(f"Error while encoding audio to AAC: {e}")

        # # Decoding
        # cmd_decode = (
        #     ffmpeg.input("pipe:0", format="aac", ac=2)
        #     .output("pipe:1", format="s16le", ac=1, ar=self.sample_rate)
        #     .global_args("-loglevel", "panic")
        #     .compile()
        # )
        # try:
        #     decoded_audio_data, _ = subprocess.Popen(
        #         cmd_decode,
        #         stdin=subprocess.PIPE,
        #         stdout=subprocess.PIPE,
        #         stderr=subprocess.PIPE,
        #         bufsize=-1,
        #     ).communicate(input=aac_audio_data)
        # except Exception as e:
        #     raise Exception(f"Error while decoding AAC audio data: {e}")

        # Encoding
        try:
            aac_audio_data, _ = (
                ffmpeg.input("pipe:0", format="s16le", ac=1, ar=self.sample_rate)
                .output("pipe:1", format="adts", ac=2, b=f"{bitrate}k")
                .global_args("-loglevel", "panic")
                .run(input=audio_data_bytes, capture_stdout=True, capture_stderr=True)
            )
        except Exception as e:
            raise Exception(f"Error while encoding audio to AAC: {e}")

        # Decoding
        try:
            decoded_audio_data, _ = (
                ffmpeg.input("pipe:0", format="aac", ac=2)
                .output("pipe:1", format="s16le", ac=1, ar=self.sample_rate)
                .global_args("-loglevel", "panic")
                .run(input=aac_audio_data, capture_stdout=True, capture_stderr=True)
            )
        except Exception as e:
            raise Exception(f"Error while decoding AAC audio data: {e}")

        decoded_audio = (np.frombuffer(decoded_audio_data, np.int16) / 32768.0).astype(np.float32)

        # Find alignment offset
        # offset = self.find_alignment_offset(audio_data, decoded_audio)
        # print("AAC offset: ", offset)
        offset = self.dalay

        # Align the audio using the offset
        # if offset > 0:
        # aligned_audio = np.concatenate((np.zeros(offset), decoded_audio))
        aligned_audio = decoded_audio[offset:]
        # else:
        #     aligned_audio = decoded_audio[:-offset]

        # Ensure the output audio has the same length as the input
        aligned_audio = aligned_audio[: len(audio_data)]

        return aligned_audio


@jit(nopython=True)
def set_holes(
    spectra,
    holes_num,
    holes_width_min_freq,
    holes_width_max_freq,
    holes_width_min_time,
    holes_width_max_time,
    cutoff_index,
):
    for _ in range(holes_num):
        freq_idx = np.random.randint(0, cutoff_index + 1)
        time_idx = np.random.randint(0, spectra.shape[1])
        hole_width_time = np.random.randint(holes_width_min_time, holes_width_max_time + 1)
        hole_width_freq = np.random.randint(holes_width_min_freq, holes_width_max_freq + 1)
        spectra[
            freq_idx - hole_width_freq : freq_idx + hole_width_freq,
            time_idx - hole_width_time : time_idx + hole_width_time,
        ] = 0
    return spectra


class SpectralTimeFreqHolesPerturb:
    def __init__(
        self,
        sample_rate,
        stft_frame_length=1024,
        stft_frame_step=256,
        holes_num_min=1,
        holes_num_max=250,
        holes_width_min_freq=1,
        holes_width_max_freq=9,
        holes_width_min_time=1,
        holes_width_max_time=12,
        cutoff_freq=10000,
    ):
        self._sample_rate = sample_rate
        self._stft_frame_length = stft_frame_length
        self._stft_frame_step = stft_frame_step
        self._holes_num_min = holes_num_min
        self._holes_num_max = holes_num_max
        self._holes_width_min_freq = holes_width_min_freq
        self._holes_width_max_freq = holes_width_max_freq
        self._holes_width_min_time = holes_width_min_time
        self._holes_width_max_time = holes_width_max_time
        self._cutoff_freq = cutoff_freq

    def __call__(self, data):
        return self.process(data)

    def process(self, data):
        spectra = librosa.stft(
            data, n_fft=self._stft_frame_length, hop_length=self._stft_frame_step
        )
        cutoff_index = int(self._cutoff_freq * self._stft_frame_length / self._sample_rate)
        holes_num = np.random.randint(self._holes_num_min, self._holes_num_max + 1)
        spectra = set_holes(
            spectra,
            holes_num,
            self._holes_width_min_freq,
            self._holes_width_max_freq,
            self._holes_width_min_time,
            self._holes_width_max_time,
            cutoff_index,
        )
        return librosa.istft(
            spectra,
            n_fft=self._stft_frame_length,
            hop_length=self._stft_frame_step,
            length=len(data),
        )
