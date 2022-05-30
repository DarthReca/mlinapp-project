from typing import Optional

from torch.utils.data import DataLoader
from torchvision.datasets import CelebA
from pytorch_lightning import LightningDataModule
import torchvision.transforms as T


class CelebADataModule(LightningDataModule):
    def __init__(self, batch_size: int):
        super().__init__()
        self.transform = T.ToTensor()
        self.val_dataset = None
        self.train_dataset = None
        self.test_dataset = None
        self.batch_size = batch_size

    def prepare_data(self) -> None:
        CelebA(root="data/images", split="all", download=True)

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit" or stage is None:
            self.train_dataset = CelebA("data", split="train", transform=self.transform)
            self.val_dataset = CelebA("data", split="valid", transform=self.transform)
        if stage == "test" or stage is None:
            self.test_dataset = CelebA("data", split="test", transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)
