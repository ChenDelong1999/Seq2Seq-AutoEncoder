import os
import subprocess
from multiprocessing import Pool

def extract(i):
    file = f"sa_{i:06}.tar"
    dir = os.path.join(parent, f"sa_{i:06}")
    if os.path.exists(file):
        print(f"Extracting {file} to {dir}")
        os.makedirs(dir, exist_ok=True)
        # subprocess.call(["tar", "-xf", file, "-C", dir])
        subprocess.call(["tar", "--skip-old-files", "-xf", file, "-C", dir])
        print(f"Extracted {file}")
    else:
        print(f"File {file} does not exist")

if __name__ == "__main__":
    # os.chdir("/cpfs/shared/research-llm/instruc_data_en/multimodal_instruct_tuning/sa1b/SA-1B/raw")
    # parent = "/cpfs/shared/research-llm/instruc_data_en/multimodal_instruct_tuning/sa1b/SA-1B/EXTRACTED"
    os.chdir("/home/dchenbs/workspace/datasets/sa1b/raw")
    parent = "/home/dchenbs/workspace/datasets/sa1b/"
    with Pool() as p:
        p.map(extract, range(50,51))