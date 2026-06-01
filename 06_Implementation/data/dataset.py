"""
PyTorch Dataset for Dental Fluorosis (DF) and Skeletal Fluorosis (SF).

DF: 200 intraoral photos, 4-class, 50/class, 512x256 PNG
SF: 80 X-ray images, 4-class (N21/M34/Mo13/S12), 512x1024 PNG
"""
import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T


class DFDataset(Dataset):
    """Dental Fluorosis dataset. Expects data/dental_fluorosis/images/{grade}/ layout."""

    GRADE_MAP = {"normal": 0, "mild": 1, "moderate": 2, "severe": 3}
    GRADE_NAMES = {0: "Normal", 1: "Mild", 2: "Moderate", 3: "Severe"}

    def __init__(self, root, split="train", transform=None, split_seed=42):
        self.root = root
        self.split = split
        self.transform = transform
        self.samples = []

        for grade_name, grade_id in self.GRADE_MAP.items():
            img_dir = None
            for candidate in [
                os.path.join(root, "data", "dental_fluorosis", "images", grade_name),
                os.path.join(root, "dental_fluorosis", "images", grade_name),  # Kaggle
                os.path.join(root, "images", grade_name),
            ]:
                if os.path.isdir(candidate):
                    img_dir = candidate
                    break
            if img_dir is None:
                raise FileNotFoundError(f"DF images not found for grade '{grade_name}' under root={root}")
            files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
            for fname in files:
                self.samples.append({
                    "path": os.path.join(img_dir, fname),
                    "grade": grade_id,
                    "grade_name": grade_name,
                })

        self._split(split_seed)

    def _split(self, seed):
        np.random.seed(seed)
        indices = np.random.permutation(len(self.samples))
        n_total = len(self.samples)
        n_train, n_val = int(n_total * 0.6), int(n_total * 0.1)

        if self.split == "train":
            self.samples = [self.samples[i] for i in indices[:n_train]]
        elif self.split == "val":
            self.samples = [self.samples[i] for i in indices[n_train:n_train + n_val]]
        elif self.split == "test":
            self.samples = [self.samples[i] for i in indices[n_train + n_val:]]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = Image.open(s["path"]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, s["grade"]


class SFDataset(Dataset):
    """Skeletal Fluorosis dataset. Uses GT.xlsx for labels.
    
    Supports both hard labels (majority vote) and soft labels (5-rater average).
    Set use_soft_labels=True to get soft probability vectors instead of integer labels.
    """

    GRADE_NAMES = {0: "Normal", 1: "Mild", 2: "Moderate", 3: "Severe"}

    def __init__(self, root, split="train", transform=None, use_soft_labels=False):
        self.root = root
        self.split = split
        self.transform = transform
        self.use_soft_labels = use_soft_labels

        gt_path = None
        for candidate in [
            os.path.join(root, "data", "skeletal_fluorosis", "GT.xlsx"),
            os.path.join(root, "skeletal_fluorosis", "GT.xlsx"),  # Kaggle
            os.path.join(root, "GT.xlsx"),
        ]:
            if os.path.exists(candidate):
                gt_path = candidate
                break
        if gt_path is None:
            raise FileNotFoundError(f"GT.xlsx not found under root={root}")
        df = pd.read_excel(gt_path, sheet_name="Sheet1")
        self.labels = {}
        self.soft_labels = {}
        self.parts = {}
        self.splits = {}
        
        # Try to load 5-rater annotations for soft labels
        rater_columns = []
        for _, row in df.iterrows():
            fid = int(row["ID"])
            self.labels[fid] = int(row["Multiple"])
            self.parts[fid] = row["Part"]
            self.splits[fid] = row["Mode"]
        
        # Look for individual rater columns (R1-R5, Rater1-Rater5, etc.)
        rater_col_patterns = [
            ["R1", "R2", "R3", "R4", "R5"],
            ["Rater1", "Rater2", "Rater3", "Rater4", "Rater5"],
            ["Reader1", "Reader2", "Reader3", "Reader4", "Reader5"],
        ]
        for pattern in rater_col_patterns:
            if all(c in df.columns for c in pattern):
                rater_columns = pattern
                break
        
        if rater_columns:
            # Build soft labels from rater votes
            for _, row in df.iterrows():
                fid = int(row["ID"])
                votes = [int(row[c]) for c in rater_columns]
                # Create probability vector from rater votes
                soft = np.zeros(4, dtype=np.float32)
                for v in votes:
                    if 0 <= v < 4:
                        soft[v] += 1.0
                soft /= soft.sum()
                self.soft_labels[fid] = soft
            print(f"[SFDataset] Loaded 5-rater soft labels from columns: {rater_columns}")
        else:
            # Fallback: try loading from separate Excel files
            rater_dir = None
            for candidate in [
                os.path.join(root, "data", "skeletal_fluorosis", "raters"),
                os.path.join(root, "skeletal_fluorosis", "raters"),
            ]:
                if os.path.isdir(candidate):
                    rater_dir = candidate
                    break
            
            if rater_dir:
                rater_files = sorted([f for f in os.listdir(rater_dir) if f.endswith('.xlsx')])
                if len(rater_files) >= 3:
                    all_ratings = {}
                    for rf in rater_files[:5]:
                        rater_df = pd.read_excel(os.path.join(rater_dir, rf))
                        for _, row in rater_df.iterrows():
                            fid = int(row["ID"])
                            grade = int(row["Grade"]) if "Grade" in row else int(row.iloc[1])
                            if fid not in all_ratings:
                                all_ratings[fid] = []
                            all_ratings[fid].append(grade)
                    
                    for fid, ratings in all_ratings.items():
                        soft = np.zeros(4, dtype=np.float32)
                        for r in ratings:
                            if 0 <= r < 4:
                                soft[r] += 1.0
                        soft /= soft.sum()
                        self.soft_labels[fid] = soft
                    print(f"[SFDataset] Loaded {len(rater_files)}-rater soft labels from {rater_dir}")
        
        if not self.soft_labels and use_soft_labels:
            print("[SFDataset] WARNING: use_soft_labels=True but no rater data found. Falling back to hard labels.")
            self.use_soft_labels = False

        for candidate in [
            os.path.join(root, "data", "skeletal_fluorosis", "images"),
            os.path.join(root, "skeletal_fluorosis", "images"),  # Kaggle
            os.path.join(root, "images"),
        ]:
            if os.path.isdir(candidate):
                self.img_dir = candidate
                break

        self.samples = []
        for fid in sorted(self.labels.keys()):
            if self.split == "test" and self.splits[fid] == "test":
                self.samples.append(fid)
            elif self.split == "val" and self.splits.get(fid) == "train" and fid % 7 == 0:
                self.samples.append(fid)
            elif self.split == "train" and self.splits.get(fid) == "train" and fid % 7 != 0:
                self.samples.append(fid)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fid = self.samples[idx]
        img = Image.open(os.path.join(self.img_dir, f"{fid}.png")).convert("RGB")
        if self.transform:
            img = self.transform(img)
        if self.use_soft_labels and fid in self.soft_labels:
            return img, torch.tensor(self.soft_labels[fid], dtype=torch.float32)
        return img, self.labels[fid]


def get_transforms(task="df", is_train=True):
    """Standard transforms for DF and SF."""
    if task == "df":
        input_size = 224
        if is_train:
            return T.Compose([
                T.RandomResizedCrop(input_size, scale=(0.8, 1.0)),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomRotation(10),
                T.ColorJitter(brightness=0.2, contrast=0.2),
                T.RandAugment(num_ops=2, magnitude=9),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            return T.Compose([
                T.Resize((input_size, input_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
    else:
        input_size = 224
        if is_train:
            return T.Compose([
                T.RandomResizedCrop(input_size, scale=(0.8, 1.0)),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomRotation(10),
                T.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),  # X-ray specific
                T.RandAugment(num_ops=2, magnitude=9),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            return T.Compose([
                T.Resize((input_size, input_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])


def create_dataloaders(root, task="df", batch_size=32, num_workers=4, use_soft_labels=False):
    """Create train/val/test dataloaders."""
    if task == "df":
        train_ds = DFDataset(root, split="train", transform=get_transforms(task, is_train=True))
        val_ds = DFDataset(root, split="val", transform=get_transforms(task, is_train=False))
        test_ds = DFDataset(root, split="test", transform=get_transforms(task, is_train=False))
    else:
        train_ds = SFDataset(root, split="train", transform=get_transforms(task, is_train=True), use_soft_labels=use_soft_labels)
        val_ds = SFDataset(root, split="val", transform=get_transforms(task, is_train=False), use_soft_labels=use_soft_labels)
        test_ds = SFDataset(root, split="test", transform=get_transforms(task, is_train=False), use_soft_labels=use_soft_labels)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader
