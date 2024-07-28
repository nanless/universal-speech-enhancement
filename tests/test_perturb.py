import os
import subprocess

import numpy as np
import soundfile as sf

from src.data.components.perturb import (
    AACConversionPerturb,
    BandRejectPerturb,
    BitCrushPerturb,
    ColoredNoisePerturb,
    DCOffsetPerturb,
    DRCPerturb,
    EQPerturb,
    GSMcodecsPerturb,
    LoudnessPerturb,
    LowPassPerturb,
    MP3CompressorPerturb,
    OPUSCodecsPerturb,
    PacketLossPerturb,
    PitchPerturb,
    SpeakerDistortionPerturbClipPedal,
    SpeakerDistortionPerturbHardClip,
    SpeakerDistortionPerturbPedal,
    SpeakerDistortionPerturbSigmoid1,
    SpeakerDistortionPerturbSigmoid2,
    SpeakerDistortionPerturbSoftClip,
    SpeakerDistortionPerturbSox,
    SpectralLeakagePerturb,
    SpectralTimeFreqHolesPerturb,
    SpeedPerturb,
    WhiteNoisePerturb,
)


def test_SpeedPerturb():
    temp_test_dir = "temp_test/test_SpeedPerturb"
    test_wav_path = "/mnt/d/presentpictureofnsw_01_mann_0054.wav"
    os.makedirs(temp_test_dir, exist_ok=True)
    subprocess.run(["rm", "-rf", f"{temp_test_dir}/*"], check=True)
    subprocess.run(["cp", f"{test_wav_path}", f"{temp_test_dir}"], check=True)
    data_org, sample_rate = sf.read(test_wav_path)
    for speed_rate in [0.5, 0.6, 0.7, 0.8, 0.9, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 2]:
        inst = SpeedPerturb(sample_rate, speed_rate=speed_rate)
        data = inst(data_org)[0]
        sf.write(f"{temp_test_dir}/test_SpeedPerturb_{speed_rate}.wav", data, sample_rate)


def test_PitchPerturb():
    temp_test_dir = "temp_test/test_PitchPerturb"
    test_wav_path = "/mnt/d/presentpictureofnsw_01_mann_0054.wav"
    os.makedirs(temp_test_dir, exist_ok=True)
    subprocess.run(["rm", "-rf", f"{temp_test_dir}/*"], check=True)
    subprocess.run(["cp", f"{test_wav_path}", f"{temp_test_dir}"], check=True)
    data_org, sample_rate = sf.read(test_wav_path)
    for semitone in range(-6, 7):
        inst = PitchPerturb(sample_rate, semitone=semitone - 0.5)
        data = inst(data_org)[0]
        sf.write(f"{temp_test_dir}/test_PitchPerturb_{semitone-0.5}.wav", data, sample_rate)


def test_EQPerturb():
    temp_test_dir = "temp_test/test_EQPerturb"
    test_wav_path = "/mnt/d/presentpictureofnsw_01_mann_0054.wav"
    os.makedirs(temp_test_dir, exist_ok=True)
    subprocess.run(["rm", "-rf", f"{temp_test_dir}/*"], check=True)
    subprocess.run(["cp", f"{test_wav_path}", f"{temp_test_dir}"], check=True)
    data_org, sample_rate = sf.read(test_wav_path)
    for i in range(20):
        inst = EQPerturb(sample_rate)
        data = inst(data_org)
        sf.write(f"{temp_test_dir}/test_EQPerturb_{i}.wav", data, sample_rate)


def test_DRCPerturb():
    temp_test_dir = "temp_test/test_DRCPerturb"
    test_wav_path = "/mnt/d/presentpictureofnsw_01_mann_0054.wav"
    os.makedirs(temp_test_dir, exist_ok=True)
    subprocess.run(["rm", "-rf", f"{temp_test_dir}/*"], check=True)
    subprocess.run(["cp", f"{test_wav_path}", f"{temp_test_dir}"], check=True)
    data_org, sample_rate = sf.read(test_wav_path)
    for i in range(20):
        inst = DRCPerturb(sample_rate)
        data = inst(data_org)
        sf.write(f"{temp_test_dir}/test_DRCPerturb_{i}.wav", data, sample_rate)


def test_SpeakerDistortionPerturbSox():
    temp_test_dir = "temp_test/test_SpeakerDistortionPerturbSox"
    os.makedirs(temp_test_dir, exist_ok=True)
    test_wav_path = "/mnt/d/presentpictureofnsw_01_mann_0054.wav"
    subprocess.run(["rm", "-rf", f"{temp_test_dir}/*"], check=True)
    subprocess.run(["cp", f"{test_wav_path}", f"{temp_test_dir}"], check=True)
    data_org, sample_rate = sf.read(test_wav_path)
    for i in range(20):
        inst = SpeakerDistortionPerturbSox(sample_rate)
        data = inst(data_org)
        print(np.abs(data).max())
        sf.write(f"{temp_test_dir}/test_SpeakerDistortionPerturbSox_{i}.wav", data, sample_rate)


def test_SpeakerDistortionPerturbPedal():
    temp_test_dir = "temp_test/test_SpeakerDistortionPerturbPedal"
    os.makedirs(temp_test_dir, exist_ok=True)
    test_wav_path = "/mnt/d/presentpictureofnsw_01_mann_0054.wav"
    subprocess.run(["rm", "-rf", f"{temp_test_dir}/*"], check=True)
    subprocess.run(["cp", f"{test_wav_path}", f"{temp_test_dir}"], check=True)
    data_org, sample_rate = sf.read(test_wav_path)
    for i in range(20):
        inst = SpeakerDistortionPerturbPedal(sample_rate)
        data = inst(data_org)
        print(np.abs(data).max())
        sf.write(f"{temp_test_dir}/test_SpeakerDistortionPerturbPedal_{i}.wav", data, sample_rate)


def test_SpeakerDistortionPerturbClipPedal():
    temp_test_dir = "temp_test/test_SpeakerDistortionPerturbClipPedal"
    os.makedirs(temp_test_dir, exist_ok=True)
    test_wav_path = "/mnt/d/presentpictureofnsw_01_mann_0054.wav"
    subprocess.run(["rm", "-rf", f"{temp_test_dir}/*"], check=True)
    subprocess.run(["cp", f"{test_wav_path}", f"{temp_test_dir}"], check=True)
    data_org, sample_rate = sf.read(test_wav_path)
    for i in range(20):
        inst = SpeakerDistortionPerturbClipPedal(sample_rate)
        data = inst(data_org)
        sf.write(
            f"{temp_test_dir}/test_SpeakerDistortionPerturbClipPedal_{i}.wav", data, sample_rate
        )


def test_SpeakerDistortionPerturbHardClip():
    temp_test_dir = "temp_test/test_SpeakerDistortionPerturbHardClip"
    os.makedirs(temp_test_dir, exist_ok=True)
    test_wav_path = "/mnt/d/presentpictureofnsw_01_mann_0054.wav"
    subprocess.run(["rm", "-rf", f"{temp_test_dir}/*"], check=True)
    subprocess.run(["cp", f"{test_wav_path}", f"{temp_test_dir}"], check=True)
    data_org, sample_rate = sf.read(test_wav_path)
    for i in range(20):
        inst = SpeakerDistortionPerturbHardClip(sample_rate)
        data = inst(data_org)
        print(np.abs(data).max())
        sf.write(
            f"{temp_test_dir}/test_SpeakerDistortionPerturbHardClip_{i}.wav", data, sample_rate
        )


def test_SpeakerDistortionPerturbSoftClip():
    temp_test_dir = "temp_test/test_SpeakerDistortionPerturbSoftClip"
    os.makedirs(temp_test_dir, exist_ok=True)
    test_wav_path = "/mnt/d/presentpictureofnsw_01_mann_0054.wav"
    subprocess.run(["rm", "-rf", f"{temp_test_dir}/*"], check=True)
    subprocess.run(["cp", f"{test_wav_path}", f"{temp_test_dir}"], check=True)
    data_org, sample_rate = sf.read(test_wav_path)
    for i in range(20):
        inst = SpeakerDistortionPerturbSoftClip(sample_rate)
        data = inst(data_org)
        print(np.abs(data).max())
        sf.write(
            f"{temp_test_dir}/test_SpeakerDistortionPerturbSoftClip_{i}.wav", data, sample_rate
        )


def test_SpeakerDistortionPerturbSigmoid1():
    temp_test_dir = "temp_test/test_SpeakerDistortionPerturbSigmoid1"
    os.makedirs(temp_test_dir, exist_ok=True)
    test_wav_path = "/mnt/d/presentpictureofnsw_01_mann_0054.wav"
    subprocess.run(["rm", "-rf", f"{temp_test_dir}/*"], check=True)
    subprocess.run(["cp", f"{test_wav_path}", f"{temp_test_dir}"], check=True)
    data_org, sample_rate = sf.read(test_wav_path)
    for i in range(20):
        inst = SpeakerDistortionPerturbSigmoid1(sample_rate)
        data = inst(data_org)
        print(np.abs(data).max())
        sf.write(
            f"{temp_test_dir}/test_SpeakerDistortionPerturbSigmoid1_{i}.wav", data, sample_rate
        )


def test_SpeakerDistortionPerturbSigmoid2():
    temp_test_dir = "temp_test/test_SpeakerDistortionPerturbSigmoid2"
    os.makedirs(temp_test_dir, exist_ok=True)
    test_wav_path = "/mnt/d/presentpictureofnsw_01_mann_0054.wav"
    subprocess.run(["rm", "-rf", f"{temp_test_dir}/*"], check=True)
    subprocess.run(["cp", f"{test_wav_path}", f"{temp_test_dir}"], check=True)
    data_org, sample_rate = sf.read(test_wav_path)
    for i in range(20):
        inst = SpeakerDistortionPerturbSigmoid2(sample_rate)
        data = inst(data_org)
        print(np.abs(data).max())
        sf.write(
            f"{temp_test_dir}/test_SpeakerDistortionPerturbSigmoid2_{i}.wav", data, sample_rate
        )


def test_LoudnessPerturb():
    temp_test_dir = "temp_test/test_LoudnessPerturb"
    os.makedirs(temp_test_dir, exist_ok=True)
    test_wav_path = "/mnt/d/presentpictureofnsw_01_mann_0054.wav"
    subprocess.run(["rm", "-rf", f"{temp_test_dir}/*"], check=True)
    subprocess.run(["cp", f"{test_wav_path}", f"{temp_test_dir}"], check=True)
    data_org, sample_rate = sf.read(test_wav_path)
    for i in range(20):
        inst = LoudnessPerturb(sample_rate)
        data = inst(data_org)
        print(np.abs(data).max())
        sf.write(f"{temp_test_dir}/test_LoudnessPerturb_{i}.wav", data, sample_rate)


def test_LowPassPerturb():
    temp_test_dir = "temp_test/test_LowPassPerturb"
    os.makedirs(temp_test_dir, exist_ok=True)
    test_wav_path = "/mnt/d/presentpictureofnsw_01_mann_0054.wav"
    subprocess.run(["rm", "-rf", f"{temp_test_dir}/*"], check=True)
    subprocess.run(["cp", f"{test_wav_path}", f"{temp_test_dir}"], check=True)
    data_org, sample_rate = sf.read(test_wav_path)
    for i in range(20):
        inst = LowPassPerturb(sample_rate)
        data = inst(data_org)
        print(np.abs(data).max())
        sf.write(f"{temp_test_dir}/test_LowPassPerturb_{i}.wav", data, sample_rate)


def test_BandRejectPerturb():
    temp_test_dir = "temp_test/test_BandRejectPerturb"
    os.makedirs(temp_test_dir, exist_ok=True)
    test_wav_path = "/mnt/d/presentpictureofnsw_01_mann_0054.wav"
    subprocess.run(["rm", "-rf", f"{temp_test_dir}/*"], check=True)
    subprocess.run(["cp", f"{test_wav_path}", f"{temp_test_dir}"], check=True)
    data_org, sample_rate = sf.read(test_wav_path)
    for i in range(20):
        inst = BandRejectPerturb(
            sample_rate, min_center_freq=500, max_center_freq=20000, min_q=1, max_q=8
        )
        data = inst(data_org)
        print(np.abs(data).max())
        sf.write(f"{temp_test_dir}/test_BandRejectPerturb_{i}.wav", data, sample_rate)


def test_SpectralLeakagePerturb():
    temp_test_dir = "temp_test/test_SpectralLeakagePerturb"
    os.makedirs(temp_test_dir, exist_ok=True)
    test_wav_path = "/mnt/d/presentpictureofnsw_01_mann_0054.wav"
    subprocess.run(["rm", "-rf", f"{temp_test_dir}/*"], check=True)
    subprocess.run(["cp", f"{test_wav_path}", f"{temp_test_dir}"], check=True)
    data_org, sample_rate = sf.read(test_wav_path)
    for i in range(20):
        inst = SpectralLeakagePerturb(sample_rate)
        data = inst(data_org)
        sf.write(f"{temp_test_dir}/test_SpectralLeakagePerturb_{i}.wav", data, sample_rate)


def test_DCOffsetPerturb():
    temp_test_dir = "temp_test/test_DCOffsetPerturb"
    os.makedirs(temp_test_dir, exist_ok=True)
    test_wav_path = "/mnt/d/presentpictureofnsw_01_mann_0054.wav"
    subprocess.run(["rm", "-rf", f"{temp_test_dir}/*"], check=True)
    subprocess.run(["cp", f"{test_wav_path}", f"{temp_test_dir}"], check=True)
    data_org, sample_rate = sf.read(test_wav_path)
    for i in range(20):
        inst = DCOffsetPerturb(sample_rate)
        data = inst(data_org)
        sf.write(f"{temp_test_dir}/test_DCOffsetPerturb_{i}.wav", data, sample_rate)


def test_WhiteNoisePerturb():
    temp_test_dir = "temp_test/test_WhiteNoisePerturb"
    os.makedirs(temp_test_dir, exist_ok=True)
    test_wav_path = "/mnt/d/presentpictureofnsw_01_mann_0054.wav"
    subprocess.run(["rm", "-rf", f"{temp_test_dir}/*"], check=True)
    subprocess.run(["cp", f"{test_wav_path}", f"{temp_test_dir}"], check=True)
    data_org, sample_rate = sf.read(test_wav_path)
    for i in range(20):
        inst = WhiteNoisePerturb(sample_rate, snr_min=0, snr_max=50)
        data = inst(data_org)
        sf.write(f"{temp_test_dir}/test_WhiteNoisePerturb_{i}.wav", data, sample_rate)


def test_ColoredNoisePerturb():
    temp_test_dir = "temp_test/test_ColoredNoisePerturb"
    os.makedirs(temp_test_dir, exist_ok=True)
    test_wav_path = "/mnt/d/presentpictureofnsw_01_mann_0054.wav"
    subprocess.run(["rm", "-rf", f"{temp_test_dir}/*"], check=True)
    subprocess.run(["cp", f"{test_wav_path}", f"{temp_test_dir}"], check=True)
    data_org, sample_rate = sf.read(test_wav_path)
    for i in range(20):
        inst = ColoredNoisePerturb(sample_rate=sample_rate, snr_min=0, snr_max=50)
        data = inst(data_org)
        sf.write(f"{temp_test_dir}/test_ColoredNoisePerturb_colored_{i}.wav", data, sample_rate)


def test_OPUSCodecsPerturb():
    temp_test_dir = "temp_test/test_OPUSCodecsPerturb"
    os.makedirs(temp_test_dir, exist_ok=True)
    test_wav_path = "temp_test_wavs/rainbow_07_regular_01.wav"
    subprocess.run(["rm", "-rf", f"{temp_test_dir}/*"], check=True)
    subprocess.run(["cp", f"{test_wav_path}", f"{temp_test_dir}"], check=True)
    data_org, sample_rate = sf.read(test_wav_path)
    print("sample rate:", sample_rate)
    for i in range(50):
        inst = OPUSCodecsPerturb(sample_rate=sample_rate)
        data = inst(data_org)
        sf.write(f"{temp_test_dir}/test_OPUSCodecsPerturb_{i}.wav", data, sample_rate)


def test_GSMcodecsPerturb():
    temp_test_dir = "temp_test/test_GSMcodecsPerturb"
    os.makedirs(temp_test_dir, exist_ok=True)
    test_wav_path = "temp_test_wavs/rainbow_07_regular_01.wav"
    subprocess.run(["rm", "-rf", f"{temp_test_dir}/*"], check=True)
    subprocess.run(["cp", f"{test_wav_path}", f"{temp_test_dir}"], check=True)
    data_org, sample_rate = sf.read(test_wav_path)
    for i in range(20):
        inst = GSMcodecsPerturb(sample_rate=sample_rate)
        data = inst(data_org)
        sf.write(f"{temp_test_dir}/test_GSMcodecsPerturb_{i}.wav", data, sample_rate)


def test_MP3CompressorPerturb():
    temp_test_dir = "temp_test/test_MP3CompressorPerturb"
    os.makedirs(temp_test_dir, exist_ok=True)
    test_wav_path = "temp_test_wavs/rainbow_07_regular_01.wav"
    subprocess.run(["rm", "-rf", f"{temp_test_dir}/*"], check=True)
    subprocess.run(["cp", f"{test_wav_path}", f"{temp_test_dir}"], check=True)
    data_org, sample_rate = sf.read(test_wav_path)
    for i in range(20):
        inst = MP3CompressorPerturb(sample_rate=sample_rate)
        data = inst(data_org)
        sf.write(f"{temp_test_dir}/test_MP3CompressorPerturb_{i}.wav", data, sample_rate)


def test_BitCrushPerturb():
    temp_test_dir = "temp_test/test_BitCrushPerturb"
    os.makedirs(temp_test_dir, exist_ok=True)
    test_wav_path = "/mnt/d/presentpictureofnsw_01_mann_0054.wav"
    subprocess.run(["rm", "-rf", f"{temp_test_dir}/*"], check=True)
    subprocess.run(["cp", f"{test_wav_path}", f"{temp_test_dir}"], check=True)
    data_org, sample_rate = sf.read(test_wav_path)
    for i in range(20):
        inst = BitCrushPerturb(sample_rate=sample_rate)
        data = inst(data_org)
        sf.write(f"{temp_test_dir}/test_BitCrushPerturb_{i}.wav", data, sample_rate)


def test_PacketLossPerturb():
    temp_test_dir = "temp_test/test_PacketLossPerturb"
    os.makedirs(temp_test_dir, exist_ok=True)
    test_wav_path = "/mnt/d/presentpictureofnsw_01_mann_0054.wav"
    subprocess.run(["rm", "-rf", f"{temp_test_dir}/*"], check=True)
    subprocess.run(["cp", f"{test_wav_path}", f"{temp_test_dir}"], check=True)
    data_org, sample_rate = sf.read(test_wav_path)
    for i in range(20):
        inst = PacketLossPerturb(sample_rate=sample_rate)
        data = inst(data_org)
        sf.write(f"{temp_test_dir}/test_PacketLossPerturb_{i}.wav", data, sample_rate)


def test_AACConversionPerturb():
    temp_test_dir = "temp_test/test_AACConversionPerturb"
    os.makedirs(temp_test_dir, exist_ok=True)
    test_wav_path = "temp_test_wavs/rainbow_07_regular_01.wav"
    subprocess.run(["rm", "-rf", f"{temp_test_dir}/*"], check=True)
    subprocess.run(["cp", f"{test_wav_path}", f"{temp_test_dir}"], check=True)
    data_org, sample_rate = sf.read(test_wav_path)
    for i in range(20):
        print("sample rate:", sample_rate)
        inst = AACConversionPerturb(sample_rate=sample_rate)
        data = inst(data_org)
        sf.write(f"{temp_test_dir}/test_AACConversionPerturb_{i}.wav", data, sample_rate)


def test_SpectralTimeFreqHolesPerturb():
    temp_test_dir = "temp_test/test_SpectralTimeFreqHolesPerturb"
    os.makedirs(temp_test_dir, exist_ok=True)
    test_wav_path = "temp_test_wavs/rainbow_07_regular_01.wav"
    subprocess.run(["rm", "-rf", f"{temp_test_dir}/*"], check=True)
    subprocess.run(["cp", f"{test_wav_path}", f"{temp_test_dir}"], check=True)
    data_org, sample_rate = sf.read(test_wav_path)
    for i in range(20):
        inst = SpectralTimeFreqHolesPerturb(sample_rate=sample_rate)
        data = inst(data_org)
        sf.write(f"{temp_test_dir}/test_SpectralTimeFreqHolesPerturb_{i}.wav", data, sample_rate)


if __name__ == "__main__":
    # test_SpeedPerturb()
    # test_PitchPerturb()
    # test_EQPerturb()
    # test_DRCPerturb()
    # test_SpeakerDistortionPerturbSox()
    # test_SpeakerDistortionPerturbPedal()
    # test_SpeakerDistortionPerturbClipPedal()
    # test_SpeakerDistortionPerturbHardClip()
    # test_SpeakerDistortionPerturbSoftClip()
    # test_SpeakerDistortionPerturbSigmoid1()
    # test_SpeakerDistortionPerturbSigmoid2()
    # # test_SpeakerDistortionPerturbMemory1()
    # test_SpeakerDistortionPerturbMemory2()
    # test_LowPassPerturb()
    # test_BandRejectPerturb()
    # test_SpectralLeakagePerturb()
    # test_DCOffsetPerturb()
    # test_ColoredNoisePerturb()
    # test_AACConversionPerturb()
    # test_PacketLossPerturb()
    # test_OPUSCodecsPerturb()
    # test_GSMcodecsPerturb()
    # test_MP3CompressorPerturb()
    test_SpectralTimeFreqHolesPerturb()
