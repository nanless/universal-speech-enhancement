import os
from glob import glob
from os import listdir, makedirs
from os.path import join

speech_dir = "/data1/data/speech/ears_dataset"
save_list_path = "/data1/data/lists/ears_all_speech.list"


# all_speakers = sorted(listdir(speech_dir))
# Define training split
# valid_speakers = ["p100", "p101"]
# test_speakers = ["p102", "p103", "p104", "p105", "p106", "p107"]

# speakers = {
#     "train": [s for s in all_speakers if s not in valid_speakers + test_speakers],
#     "valid": valid_speakers,
#     "test": test_speakers
#     }

# Hold out speaking styles
hold_out_styles = ["interjection", "melodic", "nonverbal", "vegetative"]

# Define emotions and speaking styles
emotions_styles = [
    "adoration",
    "amazement",
    "amusement",
    "anger",
    "confusion",
    "contentment",
    "cuteness",
    "desire",
    "disappointment",
    "disgust",
    "distress",
    "embarassment",
    "extasy",
    "fast",
    "fear",
    "guilt",
    "highpitch",
    "interest",
    "loud",
    "lowpitch",
    "neutral",
    "pain",
    "pride",
    "realization",
    "relief",
    "regular",
    "sadness",
    "serenity",
    "slow",
    "whisper",
]

all_speakers = sorted(listdir(speech_dir))

speech_files = []
for speaker in all_speakers:
    speech_files += sorted(glob(join(speech_dir, speaker, "*.wav")))

# Remove files of hold out styles
speech_files = [
    speech_file
    for speech_file in speech_files
    if speech_file.split("/")[-1].split("_")[0] not in hold_out_styles
]

# generate list
with open(save_list_path, "w") as f:
    for speech_file in speech_files:
        f.write(speech_file + "\n")
