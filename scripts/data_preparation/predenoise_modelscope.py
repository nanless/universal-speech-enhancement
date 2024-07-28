import multiprocessing
import os

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from tqdm import tqdm


def walk_wav_files(folder):
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith(".wav"):
                yield os.path.join(root, file)


def process_file(args):
    audio_path, original_folder, result_folder = args
    output_path = audio_path.replace(original_folder, result_folder)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    ans = pipeline(Tasks.acoustic_noise_suppression, model="damo/speech_dfsmn_ans_psm_48k_causal")
    result = ans(audio_path, output_path=output_path)


def main():
    original_folder = (
        "/root/autodl-tmp/DNS_challenge5_data/datasets_fullband/clean_fullband_MOS_filtered_240615"
    )
    result_folder = "/root/autodl-tmp/DNS_challenge5_data/datasets_fullband/clean_fullband_MOS_filtered_240615_modelscope_denoised"

    audio_paths = list(walk_wav_files(original_folder))

    with multiprocessing.Pool(processes=16) as pool:
        for _ in tqdm(
            pool.imap_unordered(
                process_file,
                [(audio_path, original_folder, result_folder) for audio_path in audio_paths],
            ),
            total=len(audio_paths),
        ):
            pass


if __name__ == "__main__":
    main()
