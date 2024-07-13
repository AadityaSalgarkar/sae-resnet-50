import argparse
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

layer_name = "layer3:112"
DATA = "ILSVRC/imagenet-1k"




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


def main_loop(index: int):
    layer_name = "layer3:" + str(index)
    print("++++++++++++++++++++++++++")
    print(layer_name)
    print("++++++++++++++++++++++++++")

    # Setup wandb logger
    wandb_logger = WandbLogger(
        project="sae-resnet50",
        name = layer_name,
        config={
            **asdict(train_config),
            **asdict(sae_config),
            "layer_name": layer_name,
            "index": index,
        },
        save_code=True,
        checkpoint_name=layer_name.replace(":", "_"),
        log_model=True,
    )

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

    # Setup checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename=f"sae-{layer_name}",
    )

    sae_trainer = SaeTrainerModule(
        model=model,
        layer_name=layer_name,
        sae_config=sae_config,
        train_config=train_config,
    )
    wandb_logger.watch(sae_trainer, log="all", log_freq=100)
    # Initialize Trainer for main training
    trainer = pl.Trainer(
        max_epochs=1,
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        log_every_n_steps=1,
        max_steps=21,
        # val_check_interval=2000,  # Run validation every 2000 training steps
        # limit_val_batches=128,  # Limit validation to 128 batches
    )
    # Train the model
    trainer.fit(sae_trainer, train_dataloader, val_dataloader)

if __name__ == "__main__":
    # Get index from command line arguments
    parser = argparse.ArgumentParser(description="Train SAE model")
    parser.add_argument("--index", type=int, required=True, help="Index for training")
    args = parser.parse_args()
    # Prepare data
    imagenet_data_train = load_dataset(DATA, split="train", trust_remote_code=True)
    imagenet_data_val = load_dataset(DATA, split="validation", trust_remote_code=True)


    # Initialize model and trainer
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    sae_config = SaeConfig()
    train_config = TrainConfig()

    print("sae_config", sae_config)
    print("train_config", train_config)

    index = args.index
    print(f"Training with index: {index}")

    main_loop(index)
