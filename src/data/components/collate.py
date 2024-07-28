import random

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence


def pad_to_longest_monaural(batch_data):
    """According to the longest item to pad dataset in one batch."""
    clean_list = []
    perturbed_list = []
    sample_num = np.zeros(len(batch_data), dtype=np.int32)
    sample_name = []
    if "sampling_rate" in batch_data[0]:
        sampling_rates = []
    if "SNR" in batch_data[0]:
        SNRs = []
    for i, sample in enumerate(batch_data):
        clean_list.append(torch.from_numpy(sample["clean"]).detach())
        perturbed_list.append(torch.from_numpy(sample["perturbed"]).detach())
        sample_num[i] = sample["clean"].shape[0]
        sample_name.append(sample["name"])
        if "sampling_rate" in sample:
            sampling_rates.append(sample["sampling_rate"])
        if "SNR" in sample:
            SNRs.append(sample["SNR"])
    clean_batch = pad_sequence(clean_list, batch_first=True, padding_value=0).detach()
    perturbed_batch = pad_sequence(perturbed_list, batch_first=True, padding_value=0).detach()
    out_batch_data = {
        "clean": clean_batch,
        "perturbed": perturbed_batch,
        "name": sample_name,
        "sample_length": torch.from_numpy(sample_num),
    }
    if "sampling_rate" in batch_data[0]:
        out_batch_data["sampling_rate"] = sampling_rates
    if "SNR" in batch_data[0]:
        out_batch_data["SNR"] = SNRs
    return out_batch_data


def pad_to_longest_monaural_inference(batch_data):
    """According to the longest item to pad dataset in one batch."""
    perturbed_list = []
    sample_num = np.zeros(len(batch_data), dtype=np.int32)
    sample_name = []
    if "sampling_rate" in batch_data[0]:
        sampling_rates = []
    if "audio_path" in batch_data[0]:
        audio_paths = []
    for i, sample in enumerate(batch_data):
        perturbed_list.append(torch.from_numpy(sample["perturbed"]).detach())
        sample_num[i] = sample["perturbed"].shape[0]
        sample_name.append(sample["name"])
        if "sampling_rate" in sample:
            sampling_rates.append(sample["sampling_rate"])
        if "audio_path" in sample:
            audio_paths.append(sample["audio_path"])
    perturbed_batch = pad_sequence(perturbed_list, batch_first=True, padding_value=0).detach()
    out_batch_data = {
        "perturbed": perturbed_batch,
        "name": sample_name,
        "sample_length": torch.from_numpy(sample_num),
    }
    if "sampling_rate" in batch_data[0]:
        out_batch_data["sampling_rate"] = sampling_rates
    if "audio_path" in batch_data[0]:
        out_batch_data["audio_path"] = audio_paths
    if "data_folder" in batch_data[0]:
        out_batch_data["data_folder"] = batch_data[0]["data_folder"]
    if "target_folder" in batch_data[0]:
        out_batch_data["target_folder"] = batch_data[0]["target_folder"]
    return out_batch_data
