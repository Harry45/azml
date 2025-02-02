import os
import torch
import torch.nn as nn
import lightning as pl
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from src import VisionTransformer

MODEL_PATH = "models/vit_pl_model.pth"

class ViTLightning(pl.LightningModule):
    def __init__(self, input_shape=(1, 28, 28),
                 num_patches=7,
                 num_blocks=2,
                 embed_dim=8,
                 num_heads=2,
                 num_classes=10,
                 lr=0.005):
        super().__init__()
        self.save_hyperparameters()

        self.model = VisionTransformer(input_shape,
                                       num_patches,
                                       num_blocks,
                                       embed_dim,
                                       num_heads,
                                       num_classes)
        self.criterion = CrossEntropyLoss()
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)

class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=128):
        super().__init__()
        self.batch_size = batch_size
        self.transform = ToTensor()

    def setup(self, stage=None):
        dataset = MNIST(root="./datasets",
                        train=True,
                        download=True,
                        transform=self.transform)
        self.train_set, self.val_set = random_split(dataset, [55000, 5000])
        self.test_set = MNIST(root="./datasets",
                              train=False,
                              download=True,
                              transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_set,
                          batch_size=self.batch_size,
                          shuffle=True)

    def val_dataloader(self):
        # val_loader = DataLoader(
        #     val_dataset,     # Your validation dataset
        #     batch_size=128,  # Adjust batch size as needed
        #     shuffle=False,
        #     num_workers=7,   # Increase this value based on CPU cores
        #     pin_memory=True  # Helps if using a GPU
        # )
        return DataLoader(self.val_set,
                          batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_set,
                          batch_size=self.batch_size)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = ViTLightning()
    data_module = MNISTDataModule()

    trainer = pl.Trainer(
        max_epochs=5,
        accelerator=device,
        log_every_n_steps=10
    )

    if not os.path.exists(MODEL_PATH):
        print("\nTraining model...")
        trainer.fit(model, data_module)
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"\nModel saved to {MODEL_PATH}")
    else:
        print(f"\nLoading model from {MODEL_PATH}...")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

    print("\nEvaluating model...")
    trainer.test(model, datamodule=data_module)

if __name__ == "__main__":
    main()