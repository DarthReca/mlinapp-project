from typing import Optional

import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CelebA
from pytorch_lightning import LightningDataModule
import torchvision.transforms as T
import numpy as np


class CelebADataModule(LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        num_workers: int = 4,
        image_size: int = 128,
        indices_file: Optional[str] = None,
    ):
        super().__init__()
        self.transform = T.Compose(
            [
                T.CenterCrop(170),
                T.Resize(image_size),
                T.ToTensor(),
                T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        self.val_dataset = None
        self.train_dataset = None
        self.test_dataset = None
        self.batch_size = batch_size
        self.indices = np.load(indices_file) if indices_file else None
        self.num_workers = num_workers

    def prepare_data(self) -> None:
        CelebA(root="data", split="all", download=True)

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit" or stage is None:
            if self.indices is not None:
                self.train_dataset = Subset(
                    CelebA("data", split="all", transform=self.transform), self.indices
                )
            else:
                self.train_dataset = CelebA(
                    "data", split="train", transform=self.transform
                )
            self.val_dataset = Subset(
                CelebA("data", split="valid", transform=self.transform),
                torch.arange(10),
            )
        if stage == "test" or stage is None:
            self.test_dataset = CelebA("data", split="test", transform=self.transform)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, num_workers=self.num_workers, batch_size=1, shuffle=False
        )
