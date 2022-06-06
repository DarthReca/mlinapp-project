from typing import Optional

from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CelebA
from pytorch_lightning import LightningDataModule
import torchvision.transforms as T
import numpy as np


class CelebADataModule(LightningDataModule):
    def __init__(self, batch_size: int, indices_file: Optional[str] = None):
        super().__init__()
        self.transform = T.ToTensor()
        self.train_dataset = None
        self.test_dataset = None
        self.batch_size = batch_size
        self.indices = np.load(indices_file) if indices_file else None

    def prepare_data(self) -> None:
        CelebA(root="data/images", split="all", download=True)

    def setup(self, stage: Optional[str] = None) -> None:
        # Filter
        if self.indices is not None:
            all_dataset = CelebA("data", split="all", transform=self.transform)
            train_lenght = len(self.indices) * 2 // 3
        # Make datasets
        if stage == "fit" or stage is None:
            if self.indices is not None:
                self.train_dataset = Subset(all_dataset, self.indices[:train_lenght])
            else:
                self.train_dataset = CelebA(
                    "data", split="train", transform=self.transform
                )
        if stage == "test" or stage is None:
            if self.indices is not None:
                self.test_dataset = Subset(all_dataset, self.indices[train_lenght:])
            else:
                self.test_dataset = CelebA(
                    "data", split="test", transform=self.transform
                )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)
