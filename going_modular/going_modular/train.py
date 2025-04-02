import os
import torch
from pathlib import Path
from torchvision import transforms
import data_setup, engine, model_builder, utils


NUM_EPOCHS = 5
BATCH_SIZE = 32
HIDDEN_UNITS = 10
LEARNING_RATE = 0.001


data_path = Path("data/")
image_path = data_path / "celebrity_face_image_dataset"
train_dir = image_path/"train"
test_dir = image_path/"test"


device = "cuda" if torch.cuda.is_available() else "cpu"


data_transform = transforms.Compose([transforms.Resize(size=(128,128)),
                                                       transforms.RandomHorizontalFlip(p=0.5),
                                                       transforms.ToTensor()])


train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    train_dir=train_dir,
    test_dir=test_dir,
    transform=data_transform,
    batch_size=BATCH_SIZE
)


model = model_builder.TinyVGG(
    input_shape=3,
    hidden_units=HIDDEN_UNITS,
    output_shape=len(class_names)
).to(device)


loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),
                             lr=LEARNING_RATE)

print(device,"device")

engine.train_and_test(model=model,
             train_dataloader=train_dataloader,
             test_dataloader=test_dataloader,
             loss_fn=loss_fn,
             optimizer=optimizer,
             epochs=NUM_EPOCHS,
             device=device)

utils.save_model(model=model,
                 target_dir="models",
                 model_name="going_modular_script_mode_tinyvgg_model.pth")