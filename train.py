import wandb
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from sae import SaeConfig, SaeTrainer, TrainConfig
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms

DATA = "ILSVRC/imagenet-1k"
# Load ImageNet dataset
from datasets import load_dataset


def custom_collate(batch):
    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]
    )
    imgs = [transform(item["image"]) for item in batch]
    targets = [item["label"] for item in batch]

    # Stack the tensors
    imgs = torch.stack(imgs)
    targets = torch.tensor(targets)

    return imgs, targets


wandb.init(project="sae-test", save_code=True)
imagenet_data = load_dataset(DATA, split="train", trust_remote_code=True)
# Create DataLoader
dataloader = DataLoader(
    imagenet_data, batch_size=8, shuffle=True, collate_fn=custom_collate
)
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
trainer = SaeTrainer(dataloader, model, "maxpool", 0, SaeConfig(), max_batches=25000)
trainer.train(num_epochs=1)
trainer.save_sae("checkpoints/sae_checkpoint.pth")
wandb.finish()
