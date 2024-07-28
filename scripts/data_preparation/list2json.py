import json
import os
import sys

import soundfile as sf
from tqdm import tqdm

if __name__ == "__main__":
    txt_file_path = sys.argv[1]
    json_file_path = sys.argv[2]
    with open(txt_file_path) as f:
        lines = f.readlines()
    with open(json_file_path, "w") as f:
        for line in tqdm(lines):
            line = line.strip()
            if os.path.exists(line):
                try:
                    data, sr = sf.read(line)
                    f.write(
                        json.dumps(
                            {
                                "file_path": line,
                                "sample_rate": sr,
                                "duration": len(data) / float(sr),
                            }
                        )
                        + "\n"
                    )
                except Exception as e:
                    print(f"Error processing {line}: {e}")
            else:
                print(f"File not found: {line}")
    print("done")
