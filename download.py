import requests
import zipfile
import tarfile
import os
from tqdm import tqdm

BASE_DATA_DIR = "./data"
TARFILE_NAME = "Task01_BrainTumour.tar"

print("Downloading data...")
response = requests.get("https://msd-for-monai.s3-us-west-2.amazonaws.com/Task01_BrainTumour.tar", stream=True)

total_size_in_bytes= int(response.headers.get('content-length', 0))
block_size = 1024
progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)

if not os.path.exists(BASE_DATA_DIR):
  os.mkdir(BASE_DATA_DIR)

tar = os.path.join(BASE_DATA_DIR, TARFILE_NAME)

with open(tar, 'wb') as file:
    for data in response.iter_content(block_size):
        progress_bar.update(len(data))
        file.write(data)

progress_bar.close()

print("Extracting download...")

with tarfile.TarFile(tar, "r") as tar_extract:
  tar_extract.extractall("data")

print("Cleaning up...")
os.remove(tar)