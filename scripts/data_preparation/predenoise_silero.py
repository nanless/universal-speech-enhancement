import os

import torch
from IPython.display import Audio, display
from omegaconf import OmegaConf
from tqdm import tqdm

original_folder = (
    "/root/autodl-tmp/DNS_challenge5_data/datasets_fullband/clean_fullband_MOS_filtered_240615"
)
result_folder = "/root/autodl-tmp/DNS_challenge5_data/datasets_fullband/clean_fullband_MOS_filtered_240615_silero_denoised"


def walk_wav_files(folder):
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith(".wav"):
                yield os.path.join(root, file)


# torch.hub.download_url_to_file('https://raw.githubusercontent.com/snakers4/silero-models/master/models.yml',
#                                'latest_silero_models.yml',
#                                progress=False)
models = OmegaConf.load("latest_silero_models.yml")

# see latest avaiable models
available_models = models.denoise_models.models
print(f"Available models {available_models}")

for am in available_models:
    _models = list(models.denoise_models.get(am).keys())
    print(f"Available models for {am}: {_models}")

name = "small_slow"  # 'large_fast', 'small_fast'
device = torch.device("cuda:0")

model, samples, utils = torch.hub.load(
    repo_or_dir="snakers4/silero-models", model="silero_denoise", name=name, device=device
)

(read_audio, save_audio, denoise) = utils

model.to(device)
for audio_path in tqdm(walk_wav_files(original_folder)):
    audio = read_audio(audio_path).to(device)
    audio_len = audio.shape[1]
    if audio.shape[1] % 48000 > 0:
        audio = torch.cat(
            [audio, torch.zeros(1, 1024 - audio.shape[1] % 1024, device=device)], dim=1
        )
    output = model(audio)
    output = output.squeeze(1)
    output = output[:, :audio_len]
    output_path = audio_path.replace(original_folder, result_folder)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_audio(output_path, output.cpu(), 48000)
