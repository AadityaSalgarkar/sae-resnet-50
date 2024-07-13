import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from sae import SaeConfig, SaeTrainerModule, TrainConfig
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms
import pytorch_lightning as pl
from datasets import load_dataset
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import random_split
from dataclasses import asdict

DATA = "ILSVRC/imagenet-1k"


"""
Layers and index
| maxpool | [1, 64, 56, 56] |
| layer1 | [1, 256, 56, 56] |
| layer2 | [1, 512, 28, 28] |
| layer3 | [1, 1024, 14, 14] |
| layer4 | [1, 2048, 7, 7] |

Example layer4:1034
"""

layer_names = ["maxpool", "layer1", "layer2", "layer3", "layer4"]
layer_max_channels = [64, 256, 512, 1024, 2048]


# Prepare data
imagenet_data_train = load_dataset(DATA, split="train", trust_remote_code=True)
imagenet_data_val = load_dataset(DATA, split="validation", trust_remote_code=True)


# Initialize model and trainer
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
sae_config = SaeConfig()
train_config = TrainConfig()
sae_trainer = SaeTrainerModule(
    model=model,
    layer_name="layer3:1012",
    sae_config=sae_config,
    train_config=train_config,
)


def custom_collate(batch):
    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize(256),
            transforms.CenterCrop(train_config.image_size),
            transforms.ToTensor(),
        ]
    )
    imgs = [transform(item["image"]) for item in batch]
    targets = [item["label"] for item in batch]

    # Stack the tensors
    imgs = torch.stack(imgs)
    targets = torch.tensor(targets)

    return imgs, targets


# dataloaders
train_dataloader = DataLoader(
    imagenet_data_train,
    batch_size=train_config.batch_size,
    shuffle=True,
    collate_fn=custom_collate,
    num_workers=train_config.num_workers,
)
val_dataloader = DataLoader(
    imagenet_data_val,
    batch_size=train_config.batch_size,
    shuffle=False,
    collate_fn=custom_collate,
    num_workers=train_config.num_workers,
)
# Setup wandb logger
wandb_logger = WandbLogger(
    project="sae-test", config=asdict(train_config), save_code=True
)

# Setup checkpoint callback
checkpoint_callback = ModelCheckpoint(
    dirpath="checkpoints",
    filename="sae-{epoch:02d}-{val_loss:.2f}",
    save_top_k=3,
    monitor="val_loss",
    mode="min",
)

# Initialize Trainer for overfitting
"""
trainer = pl.Trainer(
    max_epochs=-1,
    overfit_batches=1,
    logger=wandb_logger,
    callbacks=[checkpoint_callback],
    log_every_n_steps=1,
    check_val_every_n_epoch=128
)
"""
# Initialize Trainer for main training
trainer = pl.Trainer(
    max_epochs=1,
    logger=wandb_logger,
    callbacks=[checkpoint_callback],
    log_every_n_steps=1,
    # val_check_interval=2000,  # Run validation every 2000 training steps
    # limit_val_batches=128,  # Limit validation to 128 batches
)


# Train the model
trainer.fit(sae_trainer, train_dataloader, val_dataloader)
