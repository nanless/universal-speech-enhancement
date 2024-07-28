import numpy as np
from webrtc_audio_processing import AudioProcessingModule as AP


class WebRTCNS:
    def __init__(self, sample_rate, channels, ns_level=1):
        self.ap = AP(enable_ns=True)
        self.ap.set_stream_format(
            sample_rate, channels, sample_rate, channels
        )  # set sample rate and channels
        self.ap.set_ns_level(ns_level)  # NS level from 0 to 3
        self.frame_size = int(sample_rate * 0.01)

    def process(self, audio):
        if np.abs(audio).max() <= 1:
            audio = audio * 32768
        orig_len = len(audio)
        if len(audio) % self.frame_size != 0:
            audio = np.concatenate(
                [audio, np.zeros(self.frame_size - len(audio) % self.frame_size)]
            )
        out = np.zeros_like(audio)
        for i in range(0, len(audio), self.frame_size):
            audio_frame = audio[i : i + self.frame_size]
            audio_frame = audio_frame.astype("int16").tobytes()
            audio_out = self.ap.process_stream(audio_frame)
            audio_out = np.frombuffer(audio_out, dtype="int16").astype("float32") / 32768
            out[i : i + self.frame_size] = audio_out
        out = out[:orig_len]
        return out


class WebRTCNS_perturb:
    def __init__(self, sample_rate, channels, ns_levels):
        self.sample_rate = sample_rate
        self.channels = channels
        self.ns_levels = ns_levels
        # self.delay_min_ms = 0
        # self.delay_max_ms = 100
        if self.sample_rate == 48000:
            self.daly = 335

    # ## 使用向量内积在给定的delay_min_ms和delay_max_ms之间找到最佳的延迟
    # def find_alignment_offset(self, original, processed):
    #     delay_min = int(self.sample_rate * self.delay_min_ms / 1000)
    #     delay_max = int(self.sample_rate * self.delay_max_ms / 1000)
    #     ## 使用时域卷积计算
    #     for delay in range(delay_min, delay_max):
    #         p_1 = processed[delay:]
    #         o_1 = original[:len(p_1)]
    #         if len(p_1) > len(o_1):
    #             p_1 = p_1[:len(o_1)]
    #         corr = np.inner(o_1, p_1)
    #         if delay == delay_min:
    #             max_corr = corr
    #             max_delay = delay
    #         else:
    #             if corr > max_corr:
    #                 max_corr = corr
    #                 max_delay = delay
    #     return max_delay

    def __call__(self, audio):
        ns_level = int(np.random.choice(self.ns_levels))
        ns = WebRTCNS(self.sample_rate, self.channels, ns_level)
        audio_out = ns.process(audio)
        # delay = self.find_alignment_offset(audio, audio_out)
        # print("WebRTCNS_perturb delay: ", delay)
        return audio_out[self.daly :]


class WebRTCSAGC:
    def __init__(self, sample_rate, channels, target_level_dbfs=-3):
        self.ap = AP(agc_type=1)
        self.ap.set_stream_format(
            sample_rate, channels, sample_rate, channels
        )  # set sample rate and channels
        self.ap.set_agc_target(target_level_dbfs)  # default -3, range [-31, 0]
        self.frame_size = int(sample_rate * 0.01)

    def process(self, audio):
        if np.abs(audio).max() <= 1:
            audio = audio * 32768
        orig_len = len(audio)
        if len(audio) % self.frame_size != 0:
            audio = np.concatenate(
                [audio, np.zeros(self.frame_size - len(audio) % self.frame_size)]
            )
        out = np.zeros_like(audio)
        for i in range(0, len(audio), self.frame_size):
            audio_frame = audio[i : i + self.frame_size]
            audio_frame = audio_frame.astype("int16").tobytes()
            audio_out = self.ap.process_stream(audio_frame)
            audio_out = np.frombuffer(audio_out, dtype="int16").astype("float32") / 32768
            out[i : i + self.frame_size] = audio_out
        out = out[:orig_len]
        return out


class WebRTCSAGC_perturb:
    def __init__(self, sample_rate, channels, target_level_dbfs_list):
        self.sample_rate = sample_rate
        self.channels = channels
        self.target_level_dbfs_list = target_level_dbfs_list

    def __call__(self, audio):
        target_level_dbfs = int(np.random.choice(self.target_level_dbfs_list))
        agc = WebRTCSAGC(self.sample_rate, self.channels, target_level_dbfs)
        audio_out = agc.process(audio)
        return audio_out


# class WebRTCAGCNS:
#     def __init__(self, sample_rate, channels, target_level_dbfs=-3, ns_level=1, vad_level=1):
#         self.ap = AP(agc_type=1, enable_vad=True, enable_ns=True)

#         self.ap.set_stream_format(sample_rate, channels)      # set sample rate and channels

#         self.ap.set_agc_target(target_level_dbfs)            # default -3, range [-31, 0]

#         self.ap.set_ns_level(ns_level)                  # NS level from 0 to 3

#         self.ap.set_vad_level(vad_level)                     # VAD level from 0 to 3

#     def process(self, audio):
#         if np.abs(audio).max() <= 1:
#             audio = audio * 32768
#         orig_len = len(audio)
#         if len(audio) % self.frame_size != 0:
#             audio = np.concatenate([audio, np.zeros(self.frame_size - len(audio) % self.frame_size)])
#         out = np.zeros_like(audio)
#         for i in range(0, len(audio), self.frame_size):
#             audio_frame = audio[i:i+self.frame_size]
#             audio_frame = audio_frame.astype('int16').tobytes()
#             audio_out = self.ap.process_stream(audio_frame)
#             audio_out = np.frombuffer(audio_out, dtype='int16').astype('float32') / 32768
#             out[i:i+self.frame_size] = audio_out
#         out = out[:orig_len]
#         return out

#     def has_voice(self):
#         return self.ap.has_voice()


def test_ns():
    import matplotlib.pyplot as plt
    import numpy as np
    import soundfile as sf

    # sample_rate = 48000
    # channels = 1
    # ns_level = 3
    # audio, _ = sf.read('temp_test/test_webrtc/p267_173_noisy.wav')
    # ns = WebRTCNS(sample_rate, channels, ns_level)
    # audio_out = ns.process(audio)
    # # plt.figure()
    # # plt.subplot(211)
    # # plt.plot(audio)
    # # plt.subplot(212)
    # # plt.plot(audio_out)
    # # plt.show()
    # sf.write('temp_test/test_webrtc/p267_173_noisy_ns.wav', audio_out, sample_rate)
    # sample_rate = 16000
    # channels = 1
    # ns_level = 3
    # audio, _ = sf.read('temp_test/test_webrtc/p226_039_noisy_16k.wav')
    # ns = WebRTCNS(sample_rate, channels, ns_level)
    # audio_out = ns.process(audio)
    # # plt.figure()
    # # plt.subplot(211)
    # # plt.plot(audio)
    # # plt.subplot(212)
    # # plt.plot(audio_out)
    # # plt.show()
    # sf.write('temp_test/test_webrtc/p226_039_noisy_16k_ns.wav', audio_out, sample_rate)

    sample_rate = 48000
    channels = 1
    ns_level = 3

    audio, _ = sf.read("/data2/zhounan/data/speech/wav48k/VCTK-Corpus/wav48/p225/p225_001.wav")

    ns = WebRTCNS(sample_rate, channels, ns_level)

    audio_out = ns.process(audio)

    # plt.figure()
    # plt.subplot(211)
    # plt.plot(audio)
    # plt.subplot(212)
    # plt.plot(audio_out)
    # plt.show()

    sf.write("p225_001_ns.wav", audio_out, sample_rate)


def test_agc():
    import matplotlib.pyplot as plt
    import numpy as np
    import soundfile as sf

    sample_rate = 48000
    channels = 1
    target_level_dbfs = -3

    audio, _ = sf.read("temp_test/test_webrtc/p267_173_noisy.wav")

    agc = WebRTCSAGC(sample_rate, channels, target_level_dbfs)

    audio_out = agc.process(audio)

    # plt.figure()
    # plt.subplot(211)
    # plt.plot(audio)
    # plt.subplot(212)
    # plt.plot(audio_out)
    # plt.show()

    sf.write("temp_test/test_webrtc/p267_173_noisy_agc.wav", audio_out, sample_rate)

    sample_rate = 48000
    channels = 1
    target_level_dbfs = -3

    audio, _ = sf.read("temp_test/test_webrtc/p267_173_clean.wav")

    agc = WebRTCSAGC(sample_rate, channels, target_level_dbfs)

    audio_out = agc.process(audio)

    # plt.figure()
    # plt.subplot(211)
    # plt.plot(audio)
    # plt.subplot(212)
    # plt.plot(audio_out)
    # plt.show()

    sf.write("temp_test/test_webrtc/p267_173_clean_agc.wav", audio_out, sample_rate)

    sample_rate = 16000
    channels = 1
    target_level_dbfs = -3

    audio, _ = sf.read("temp_test/test_webrtc/p226_039_noisy_16k.wav")

    agc = WebRTCSAGC(sample_rate, channels, target_level_dbfs)

    audio_out = agc.process(audio)

    # plt.figure()
    # plt.subplot(211)
    # plt.plot(audio)
    # plt.subplot(212)
    # plt.plot(audio_out)
    # plt.show()

    sf.write("temp_test/test_webrtc/p226_039_noisy_16k_agc.wav", audio_out, sample_rate)


if __name__ == "__main__":
    test_ns()
    # test_agc()
