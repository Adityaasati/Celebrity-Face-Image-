
import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import customs


NUM_WORKERS = os.cpu_count()



def create_dataloaders(
    train_dir: str,
    test_dir: str,
    transform: transforms.Compose,
    batch_size: int,
    num_workers: int=NUM_WORKERS
):
  train_data = customs.ImageFolderCustom(train_dir, transform=transform)
  test_data = customs.ImageFolderCustom(test_dir, transform=transform)

  # Get class names
  class_names = train_data.classes

  # Turn images into data loaders
  train_custom_dataloader = DataLoader(
      train_data,
      batch_size=batch_size,
      shuffle=True,
      num_workers=num_workers,
      pin_memory=True,
  )
  test_custom_dataloader = DataLoader(
      test_data,
      batch_size=batch_size,
      shuffle=False,
      num_workers=num_workers,
      pin_memory=True,
  )

  return train_custom_dataloader, test_custom_dataloader, class_names