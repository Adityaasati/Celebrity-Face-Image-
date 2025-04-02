import requests
import zipfile
import os
from pathlib import Path
import random
import shutil


data_path = Path("data/")
image_path = data_path / "celebrity_face_image_dataset"



if image_path.is_dir():
    print(f"{image_path} directory already exists... skipping download")
else:
    print(f"{image_path} does not exist, creating one...")
    image_path.mkdir(parents=True, exist_ok=True)


dataset_url = "https://github.com/Adityaasati/PyTorch-Face-Recognition/raw/refs/heads/main/celebrity_face_image_dataset.zip"
with open(data_path / "celebrity_face_image_dataset.zip", "wb") as f:
    request = requests.get(dataset_url)
    print("Downloading Celebrities Face Image data...")
    f.write(request.content)

with zipfile.ZipFile(data_path / "celebrity_face_image_dataset.zip", "r") as zip_ref:
    print("Unzipping Celebrities Face Image data...")
    zip_ref.extractall(image_path)



