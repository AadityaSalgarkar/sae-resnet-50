import torch
import wandb
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import resnet50, ResNet50_Weights
from tqdm import tqdm
import datasets


from .sae import Sae
from .config import SaeConfig


try:
    from bitsandbytes.optim import Adam8bit as Adam

    print("Using 8-bit Adam from bitsandbytes")
except ImportError:
    from torch.optim import Adam

    print("bitsandbytes 8-bit Adam not available, using torch.optim.Adam")
    print("Run `pip install bitsandbytes` for less memory usage.")


class SaeTrainer:
    def __init__(
        self,
        dataloader: DataLoader,
        model: nn.Module,
        layer_name: str,
        channel_index: int,
        sae_config: SaeConfig,
        max_batches: int = 10**9,
        shorten_latents: int = 32,
    ):
        self.dataloader = dataloader
        self.model = model
        self.layer_name = layer_name
        self.channel_index = channel_index
        self.sae_config = sae_config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)
        self.model.eval()

        self.sae = None
        self.optimizer = None
        self._register_hook()
        self.max_batches = max_batches
        self.shorten_latents = shorten_latents

    def _register_hook(self):
        def hook(module, input, output):
            self.activation = output.detach()[
                :, self.channel_index, : self.shorten_latents, : self.shorten_latents
            ].reshape(1, -1)

        for name, module in self.model.named_modules():
            if name == self.layer_name:
                module.register_forward_hook(hook)
                break
        else:
            raise ValueError(f"Layer {self.layer_name} not found in the model")

    def _init_sae(self):
        sample_input = next(iter(self.dataloader))[0].to(self.device)
        with torch.no_grad():
            self.model(sample_input)
        d_in = self.activation.shape[1]
        # print(f" d_in : {d_in}")
        self.sae = Sae(d_in=d_in, cfg=self.sae_config, decoder=True, device=self.device)
        # print(self.sae)
        self.optimizer = Adam(self.sae.parameters(), lr=0.0002 / (2**5))

    def train(self, num_epochs: int):
        if self.sae is None:
            self._init_sae()

        self.sae.train()
        for epoch in range(num_epochs):
            total_loss = 0
            batches_done = 0
            for batch in tqdm(self.dataloader):
                batches_done += 1
                if batches_done > self.max_batches:
                    break
                inputs = batch[0].to(self.device)
                with torch.no_grad():
                    self.model(inputs)

                self.optimizer.zero_grad()
                output = self.sae(self.activation)
                loss = torch.mean((output.sae_out - self.activation) ** 2)
                loss.backward()
                self.optimizer.step()

                wandb.log({"loss": loss.item()})
                total_loss += loss.item()

            avg_loss = total_loss / len(self.dataloader)
            # print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

    def save_sae(self, path: str):
        torch.save(self.sae.state_dict(), path)

    def load_sae(self, path: str):
        if self.sae is None:
            self._init_sae()
        self.sae.load_state_dict(torch.load(path))
