import os
import pickle
import time
from multiprocessing import Pool

import numpy as np
import pyroomacoustics as pra
from tqdm import tqdm

# 参数设置与之前类似
save_folder = "/root/autodl-tmp/data/rir/rirs_monaural_farfield_20231008_48k_alldistances_pyroomacoustics/farfield"
n_samples = 16384
n_rirs = 20000
fs = 48000
c = pra.constants.get("c")
rt60_range = [0.05, 0.8]
room_length_range = [3, 10]
room_width_range = [3, 10]
room_height_range = [2.5, 4.5]
dist_range = [0.2, 3]
Pool_num = 32  # 确保这个数字不超过你机器的核心数


def generate_rir():
    # 随机生成房间大小和 RT60
    room_dim = [
        np.random.uniform(room_length_range[0], room_length_range[1]),
        np.random.uniform(room_width_range[0], room_width_range[1]),
        np.random.uniform(room_height_range[0], room_height_range[1]),
    ]
    rt60_tgt = np.random.uniform(rt60_range[0], rt60_range[1])  # seconds

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

    # 创建鞋盒模型房间
    e_absorption, max_order = pra.inverse_sabine(rt60_tgt, room_dim)
    room = pra.ShoeBox(
        room_dim,
        fs=fs,
        materials=pra.Material(e_absorption),
        max_order=max_order,
        ray_tracing=True,
        use_rand_ism=True,
    )

    # 随机生成源和麦克风位置
    while True:
        source_loc = np.random.uniform([0.2, 0.2, 0.2], np.subtract(room_dim, 0.2))
        mic_loc = np.random.uniform([0.2, 0.2, 0.2], np.subtract(room_dim, 0.2))
        near_mic_dist = np.linalg.norm(mic_loc - source_loc)
        if dist_range[0] < near_mic_dist < dist_range[1]:
            break

    # 添加源和麦克风
    room.add_source(source_loc)
    room.add_microphone_array(pra.MicrophoneArray(mic_loc.reshape(3, 1), room.fs))

    # 计算 RIR
    room.compute_rir()
    # fig, ax = room.plot_rir()
    # fig.savefig(f"rir_{i_rir}_rt{rt60_tgt:.2f}_dist{near_mic_dist:.2f}.png")

    # 获取 RIR，转换为所需的采样数
    # h = room.rir[0][0][:n_samples]
    h = room.rir[0][0]
    h = h[np.argmax(np.abs(h)) :]
    h = h / np.max(np.abs(h))
    # h = h[:n_samples]
    return h, rt60_tgt, near_mic_dist


def gen_rir_pyroomacoustics(i_rir):
    np.random.seed(i_rir)
    start = time.time()
    save_dict = {}
    h, rt60_tgt, near_mic_dist = generate_rir()
    # import ipdb; ipdb.set_trace()

    save_dict["source_rir"] = h
    with open(
        os.path.join(save_folder, f"rir_{i_rir}_rt{rt60_tgt:.2f}_dist{near_mic_dist:.2f}.pkl"),
        "wb",
    ) as f:
        pickle.dump(save_dict, f)

    end = time.time()
    return end - start


if __name__ == "__main__":
    os.makedirs(save_folder, exist_ok=True)
    # with Pool(Pool_num) as p:
    #     rir_durations = list(tqdm(p.imap(gen_rir_pyroomacoustics, range(n_rirs)), total=n_rirs, desc='Generating RIRs'))
    # print(f"Average RIR generation time: {np.mean(rir_durations):.2f} seconds")
    for i_rir in tqdm(range(n_rirs), desc="Generating RIRs"):
        gen_rir_pyroomacoustics(i_rir)
