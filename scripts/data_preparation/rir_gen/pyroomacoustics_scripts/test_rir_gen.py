import os

import soundfile as sf
from rir_gen_monaural_48k_all_distances_farfield import generate_rir
from scipy.signal import fftconvolve

test_speech_path = (
    "/root/autodl-tmp/data/speech_48k/biaobei_standard_chinese_female_10000/009998.wav"
)
save_path = "test_rir_gen/pyroomacoustics"
os.makedirs(save_path, exist_ok=True)
speech, sr = sf.read(test_speech_path)

for i in range(20):
    rir_h, _, _ = generate_rir()
    near_rir = rir_h[:10]

    conv_speech = fftconvolve(speech, rir_h)[: len(speech)]
    conv_speech_near = fftconvolve(speech, near_rir)[: len(speech)]
    sf.write(os.path.join(save_path, f"conved_speech_{i}.wav"), conv_speech, sr)
    sf.write(os.path.join(save_path, f"conved_speech_{i}_near.wav"), conv_speech_near, sr)
