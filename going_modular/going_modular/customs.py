import random
from PIL import Image
from typing import Tuple, Dict, List
from torch.utils.data import Dataset,DataLoader
from pathlib import Path
import os
import torch

data_path = Path("data/")
image_path = data_path / "celebrity_face_image_dataset"
train_dir = image_path/"train"
test_dir = image_path/"test"
target_directory = train_dir

class_names = sorted([entry.name for entry in os.scandir(target_directory)])

def class_and_idx(target_directory):
  class_names = sorted(entry.name for entry in os.scandir(target_directory) if entry.is_dir())

  class_idx = {class_name: i for i,class_name in enumerate(class_names)}

  return class_names,class_idx

class ImageFolderCustom(Dataset):
  def __init__(self, targ_dir:str, transform=None):
    self.paths = list(Path(targ_dir).glob("*/*.jpg"))
    self.transform=transform
    self.classes, self.class_to_idx = class_and_idx(targ_dir)
    print(self.class_to_idx,"self.class_to_idx")



  def load_image(self, indx:int) -> Image:
    image_path = self.paths[indx]
    return Image.open(image_path)

  def __len__(self):
    return len(self.paths)

  def __getitem__(self, index:int) ->Tuple[torch.Tensor, int]:
    img = self.load_image(index)
    class_name = self.paths[index].parent.name
    class_idx = self.class_to_idx[class_name]

    if self.transform:
      return self.transform(img), class_idx
    else:
      return img, class_idx

