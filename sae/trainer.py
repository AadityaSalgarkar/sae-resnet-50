import torch
import wandb
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import resnet50, ResNet50_Weights
from tqdm import tqdm
import datasets
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import random_split

from .sae import Sae
from .config import SaeConfig, TrainConfig


try:
    from bitsandbytes.optim import Adam8bit as Adam

    print("Using 8-bit Adam from bitsandbytes")
except ImportError:
    from torch.optim import Adam

    print("bitsandbytes 8-bit Adam not available, using torch.optim.Adam")
    print("Run `pip install bitsandbytes` for less memory usage.")


class SaeTrainerModule(pl.LightningModule):
    def __init__(
        self,
        model,
        layer_name,
        sae_config,
        train_config,
        device="cuda",
    ):
        super().__init__()
        self.model = model.to(device)
        # Freeze the parameters of self.model
        for param in self.model.parameters():
            param.requires_grad = False
        self.device_param = torch.device(device)
        self.layer_name, self.channel_index = layer_name.split(":")
        self.channel_index = int(self.channel_index)
        self.filter_size = 14

        assert self.channel_index < self.filter_size * self.filter_size 

        self.row , self.col = self.channel_index // self.filter_size , self.channel_index % self.filter_size
        self.sae_config = sae_config
        self.sae = None
        self.activation = None
        self.train_config = train_config
        self._register_hook()
        with torch.no_grad():
            # Create a random test tensor
            batch_size = train_config.batch_size
            image_size = train_config.image_size
            num_channels = train_config.num_channels
            x_test = torch.randn(
                batch_size,
                train_config.num_channels,
                train_config.image_size,
                train_config.image_size,
            ).to(self.device_param)
            self.model(x_test)
        # Track batch with top k losses
        self.top_k_losses = []
        self.top_k = 5  # Number of top losses to track
        print("Initialization complete")

    def _register_hook(self):
        def hook(module, input, output):
            self.activation = output.detach()[:, : ,self.row , self.col].reshape(
                self.train_config.batch_size, -1
            )

        for name, module in self.model.named_modules():
            if name == self.layer_name:
                module.register_forward_hook(hook)
                break
        else:
            raise ValueError(f"Layer {self.layer_name} not found in the model")

    def _init_sae(self):
        d_in = self.activation.shape[1]
        self.sae = Sae(d_in=d_in, cfg=self.sae_config, decoder=True)
        self.sae = self.sae.to(self.device_param)
        print("sae model", self.sae)

    def forward(self, x):
        x = x.to(self.device_param)
        with torch.no_grad():
            self.model(x)
        return self.sae(self.activation)

    def training_step(self, batch, batch_idx):
        inputs, _ = batch
        output = self(inputs)
        loss = torch.mean((output.sae_out - self.activation) ** 2)
        self.log("train_loss", loss)
        self.log("auxk_loss", output.auxk_loss)
        self.log("fvu", output.fvu)
        self._update_top_k_losses(loss, batch)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, _ = batch
        output = self(inputs)
        loss = torch.mean((output.sae_out - self.activation) ** 2)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        if self.sae is None:
            self._init_sae()
        if (lr := self.train_config.lr) is None:
            d = self.sae.d_in
            lr = 0.000001
        return Adam(self.sae.parameters(), lr=lr)

    def _update_top_k_losses(self, loss, batch):
        if len(self.top_k_losses) < self.top_k:
            self.top_k_losses.append((loss.item(), batch))
            self.top_k_losses.sort(key=lambda x: x[0], reverse=True)
        elif loss.item() > self.top_k_losses[-1][0]:
            self.top_k_losses.pop()
            self.top_k_losses.append((loss.item(), batch))
            self.top_k_losses.sort(key=lambda x: x[0], reverse=True)


    def on_train_epoch_end(self):
        for i, (loss, batch) in enumerate(self.top_k_losses):
            self.log(f"top_{i+1}_loss", loss)
            # Log the image for this top loss
            images, _ = batch
            for i, image in enumerate(images):
                self.logger.experiment.log({"top_loss_image": [wandb.Image(image, caption=f"Top {i+1} Loss Image")]})
        self.top_k_losses = []  # Reset for next epoch
