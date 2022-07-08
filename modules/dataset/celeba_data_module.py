from typing import Optional

import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Subset


class CelebADataModule(LightningDataModule):
    def __init__(
        self,
        selected_attrs,  # Subset of the 40 CelebA attributes
        batch_size: int = 64,
        num_workers: int = 4,
        img_size: int = 128,
        indices_file: Optional[str] = None,
        data_root: str = "data",
        num_val_samples: int = 10,
    ):
        super().__init__()
        self.transform = T.Compose(
            [
                T.CenterCrop(170),
                T.Resize(img_size),
                T.ToTensor(),
                T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        self.val_dataset = None
        self.train_dataset = None
        self.test_dataset = None

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_val_samples = num_val_samples
        self.chosen_indices = np.load(indices_file) if indices_file else None
        self.selected_attrs = selected_attrs
        self.data_root = data_root

    def prepare_data(self) -> None:
        torchvision.datasets.CelebA(root=self.data_root, split="all", download=True)

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit" or stage is None:
            if self.chosen_indices is not None:
                self.train_dataset = Subset(
                    FilteredCelebA(
                        "all", self.transform, self.selected_attrs, self.data_root
                    ),
                    self.chosen_indices,
                )
            else:
                self.train_dataset = torchvision.datasets.CelebA(
                    self.data_root, split="train", transform=self.transform
                )
            self.val_dataset = Subset(
                FilteredCelebA(
                    "valid", self.transform, self.selected_attrs, self.data_root
                ),
                torch.arange(self.num_val_samples),
            )
        if stage == "test" or stage is None:
            self.test_dataset = (
                FilteredCelebA(
                    "test", self.transform, self.selected_attrs, self.data_root
                ),
            )
        print(
            f"Training dataset is {len(self.train_dataset)} length. Validation samples are {len(self.val_dataset)}"
        )

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
            batch_size=len(self.val_dataset),
            shuffle=False,
            drop_last=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, num_workers=self.num_workers, batch_size=1, shuffle=False
        )


class DoubleCelebADataModule(CelebADataModule):
    def __init__(
        self,
        selected_attrs,  # Subset of the 40 CelebA attributes
        generic_indices_file: str,
        target_indices_file: str,
        batch_size: int = 64,
        num_workers: int = 4,
        img_size: int = 128,
        data_root: str = "data",
        num_val_samples: int = 10,
    ):
        super().__init__(
            selected_attrs,
            batch_size,
            num_workers,
            img_size,
            generic_indices_file,
            data_root,
            num_val_samples,
        )
        self.target_indices = np.load(target_indices_file)
        self.target_train_dataset = None

    def setup(self, stage: Optional[str] = None):
        super().setup(stage)
        if stage == "fit" or stage is None:
            self.target_train_dataset = Subset(
                FilteredCelebA(
                    "all", self.transform, self.selected_attrs, self.data_root
                ),
                self.target_indices,
            )

    def train_dataloader(self):
        target_dataloader = DataLoader(
            self.target_train_dataset,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
        )
        return {"generic": super().train_dataloader(), "target": target_dataloader}


class FilteredCelebA(torchvision.datasets.CelebA):
    def __init__(self, split, transform, selected_attrs, data_root):
        super().__init__(data_root, split=split, transform=transform)
        self.selected_attrs = [self.attr_names.index(att) for att in selected_attrs]

    def __getitem__(self, index):
        img, attr = super().__getitem__(index)
        return img, attr[self.selected_attrs]


# class CelebA(Dataset):
#     def __init__(self, data_path, attr_path, mode, selected_attrs, transform):
#         """
#         Initialize the class
#         data_path: path of the images folder
#         attr_path: path of the attribute file (Format: First line with #lines, Second line: list of attributes names, then 1 line for each image with 1/-1 for each attribute)
#         mode: ['train', 'val', 'test']
#         selected_attrs: subset of the attributes we want to consider
#         transform: transformations
#         """

#         super(CelebA, self).__init__()
#         self.data_path = data_path
#         # Get the list of attributes names from the header
#         att_list = open(attr_path, "r", encoding="utf-8").readlines()[1].split()
#         # Get the column indexes of the selected attributes (sum 1 since the first column of the lines contains the image name)
#         atts = [att_list.index(att) + 1 for att in selected_attrs]
#         # Get the array of the images name (path), extracting it from the first column (not considering the first 2 lines)
#         images = np.loadtxt(attr_path, skiprows=2, usecols=[0], dtype=np.str)
#         # Get an array with #row=#images and #columns=#attributes
#         labels = np.loadtxt(attr_path, skiprows=2, usecols=atts, dtype=np.int)

#         TRAIN_IMAGES = 10  # 162770 #182000 <- used to be this one
#         VAL_IMAGES = 15  # 182637
#         TOT_IMAGES = 20  # 202599
#         # Split in train, valid and test images depending on the mode parameter
#         if mode == "train":
#             self.images = images[:TRAIN_IMAGES]
#             self.labels = labels[:TRAIN_IMAGES]
#         if mode == "val":
#             self.images = images[TRAIN_IMAGES:VAL_IMAGES]
#             self.labels = labels[TRAIN_IMAGES:VAL_IMAGES]
#         if mode == "test":
#             self.images = images[VAL_IMAGES:]
#             self.labels = labels[VAL_IMAGES:]

#         self.tf = transform

#         self.length = len(self.images)

#     def __getitem__(self, index):
#         img = self.tf(Image.open(os.path.join(self.data_path, self.images[index])))
#         att = torch.tensor(
#             (self.labels[index] + 1) // 2
#         )  # attributes values shifted from -1/1 to 0/1
#         return img, att

#     def __len__(self):
#         return self.length
