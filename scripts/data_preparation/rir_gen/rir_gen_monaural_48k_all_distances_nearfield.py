import os
import pickle
import time

# import scipy.signal as ss
# import soundfile as sf
from multiprocessing import Pool

import numpy as np
import rir_generator as rir
from tqdm import tqdm

# signal, fs = sf.read("../../../data/speech/cleansets_16k_20220819/hi_fi_tts_v0/speech/bambatse_09_haggard_0133.wav")

save_folder = (
    "/root/autodl-tmp/data/rir/rirs_monaural_farfield_20240531_48k_alldistances/nearfield"
)
# save_folder = "/data2/zhounan/data/rir/rirs_monaural_farfield_20231108_48k_alldistances/nearfield"
n_samples = 16384
n_rirs = 10000
fs = 48000
c = 340
rt60_range = [0.05, 0.25]
room_length_range = [3, 10]
room_width_range = [3, 10]
room_height_range = [2.5, 4.5]
dist_range = [0.02, 0.3]
Pool_num = 1


def gen_one_pair(i_rir):
    np.random.seed(i_rir)
    start = time.time()
    save_dict = {}

    room_dim = [
        np.random.uniform(room_length_range[0], room_length_range[1]),
        np.random.uniform(room_width_range[0], room_width_range[1]),
        np.random.uniform(room_height_range[0], room_height_range[1]),
    ]

    L = np.array(room_dim)
    # Volume of room
    V = np.prod(L)
    # Surface area of walls
    A = L[::-1] * np.roll(L[::-1], 1)
    S = 2 * np.sum(A)
    while True:
        rt60_tgt = np.random.uniform(rt60_range[0], rt60_range[1])  # seconds
        alpha = 24 * np.log(10.0) * V / (c * S * rt60_tgt)
        if alpha < 1:
            break

    source_x = np.random.uniform(0.5, room_dim[0] - 0.5)
    source_y = np.random.uniform(0.5, room_dim[1] - 0.5)
    source_z = np.random.uniform(1, 2.3)
    source_loc = [source_x, source_y, source_z]

    while True:
        near_mic_loc_x = np.random.uniform(source_loc[0] - 0.3, source_loc[0] + 0.3)
        near_mic_loc_x = min(max(near_mic_loc_x, 0.1), room_dim[0] - 0.1)
        near_mic_loc_y = np.random.uniform(source_loc[1] - 0.3, source_loc[1] + 0.3)
        near_mic_loc_y = min(max(near_mic_loc_y, 0.1), room_dim[1] - 0.1)
        near_mic_loc_z = np.random.uniform(source_loc[2] - 0.3, source_loc[2] + 0.3)
        near_mic_loc_z = min(max(near_mic_loc_z, 0.1), room_dim[2] - 0.1)
        near_mic_loc = [near_mic_loc_x, near_mic_loc_y, near_mic_loc_z]
        near_mic_dist = np.linalg.norm(np.array(near_mic_loc) - np.array(source_loc))
        if dist_range[0] < near_mic_dist < dist_range[1]:
            break

    h = rir.generate(
        c=c,  # Sound velocity (m/s)
        fs=fs,  # Sample frequency (samples/s)
        r=near_mic_loc,  # Receiver position(s) [x y z] (m)
        s=source_loc,  # Source position [x y z] (m)
        L=room_dim,  # Room dimensions [x y z] (m)
        reverberation_time=rt60_tgt,  # Reverberation time (s)
        nsample=n_samples,  # Number of output samples,
        mtype=rir.mtype.omnidirectional,  # Type of microphone
        order=-1,  # -1 equals maximum reflection order!
        dim=3,  # Room dimension
        orientation=None,  # Microphone orientation (rad) [azimuth elevation]
        hp_filter=True,  # Enable high-pass filter
    )

    # print(h.shape)
    # print(signal.shape)
    # h_normed = h / np.linalg.norm(h[:, 0])
    save_dict["source_rir"] = h
    with open(
        os.path.join(save_folder, f"rir_{i_rir}_rt{rt60_tgt:.2f}_dist{near_mic_dist:.2f}.pkl"),
        "wb",
    ) as f:
        pickle.dump(save_dict, f)

    end = time.time()
    return end - start

    # # Convolve 1-channel signal with 2 impulse responses
    # signal_new = ss.convolve(h_normed[:, :], signal[:, None])

    # print(signal_new.shape)
    # os.makedirs("temp_rir_examples", exist_ok=True)
    # sf.write("temp_rir_examples/original.wav", signal, fs)
    # sf.write("temp_rir_examples/convolved.wav", signal_new, fs)


if __name__ == "__main__":
    os.makedirs(save_folder, exist_ok=True)
    # for i_rir in range(n_rirs):
    #     gen_one_pair(i_rir)
    with Pool(Pool_num) as p:
        print(list(tqdm(p.imap(gen_one_pair, range(n_rirs)), total=n_rirs, desc="监视进度")))
