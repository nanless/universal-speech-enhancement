import json
import os

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.distributed as dist


class LoadWavDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        list_path=None,
        json_path=None,
        data_folder=None,
        input_json_list=None,
        input_plain_list=None,
        normalize=False,
        min_duration_seconds=None,
        max_duration_seconds=None,
        sampling_rate=None,
        output_resample=False,
        output_resample_rate=None,
        target_folder=None,
    ):
        super().__init__()
        self.list_path = list_path
        self.json_path = json_path
        self.data_folder = data_folder
        self.input_json_list = input_json_list
        self.input_plain_list = input_plain_list
        self.normalize = normalize
        self.min_duration_seconds = min_duration_seconds
        self.max_duration_seconds = max_duration_seconds
        self.sampling_rate = sampling_rate
        self.target_folder = target_folder
        self.filepaths = []
        if self.json_path:
            with open(self.json_path) as f:
                for line in f.readlines():
                    line = line.strip()
                    json_data = json.loads(line)
                    if line:
                        try:
                            if json_data["audio_filepath"] not in self.filepaths:
                                self.filepaths.append(json_data["audio_filepath"])
                        except Exception as e:
                            if json_data["file_path"] not in self.filepaths:
                                self.filepaths.append(json_data["file_path"])
        elif self.list_path:
            with open(self.list_path) as f:
                for line in f.readlines():
                    line = line.strip()
                    if line:
                        self.filepaths.append(line)
        elif self.input_json_list:
            for line in self.input_json_list:
                json_data = json.loads(line)
                if line:
                    try:
                        if json_data["audio_filepath"] not in self.filepaths:
                            self.filepaths.append(json_data["audio_filepath"])
                    except Exception as e:
                        if json_data["file_path"] not in self.filepaths:
                            self.filepaths.append(json_data["file_path"])
        elif self.input_plain_list:
            for line in self.input_plain_list:
                self.filepaths.append(line)
        elif self.data_folder:
            for root, dirs, files in os.walk(self.data_folder):
                for file in files:
                    if file.endswith(".wav"):
                        self.filepaths.append(os.path.join(root, file))
        else:
            raise ValueError("No input list provided")

        self.length = len(self.filepaths)
        self.output_resample = output_resample
        self.output_resample_rate = output_resample_rate

        if dist.is_initialized():
            self.rank = dist.get_rank()
        else:
            self.rank = 0

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        name = os.path.basename(self.filepaths[idx]).split(".wav")[0]
        perturbed_data, sr = sf.read(self.filepaths[idx])
        if perturbed_data.ndim == 2:
            perturbed_data = perturbed_data[:, 0]
        if self.sampling_rate:
            perturbed_data = librosa.resample(
                perturbed_data, orig_sr=sr, target_sr=self.sampling_rate, res_type="fft"
            )
        if self.normalize:
            perturbed_data = perturbed_data / np.max(np.abs(perturbed_data)) * 0.8
        if self.output_resample:
            perturbed_data = librosa.resample(
                perturbed_data, orig_sr=sr, target_sr=self.output_resample_rate, res_type="fft"
            )
        out = {
            "perturbed": perturbed_data.astype(np.float32),
            "name": name,
            "audio_path": self.filepaths[idx],
        }
        if self.output_resample:
            out["sampling_rate"] = self.output_resample_rate
        elif self.sampling_rate:
            out["sampling_rate"] = self.sampling_rate
        else:
            out["sampling_rate"] = sr
        if self.data_folder:
            out["data_folder"] = self.data_folder
        if self.target_folder:
            out["target_folder"] = self.target_folder
        return out
