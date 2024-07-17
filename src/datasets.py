import os
import numpy as np
import torch
from typing import Tuple
from termcolor import cprint
from sklearn.decomposition import FastICA
from src.pre_process import pre_process
from PIL import Image


class  ThingsMEGDataset(torch.utils.data.Dataset):
    def __init__(self, split: str, data_dir: str = "data") -> None:
        super().__init__()
        
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        self.split = split
        self.num_classes = 1854
        
        self.X = torch.load(os.path.join(data_dir, f"{split}_X.pt"))
        # self.X = pre_process(self.X, self.split)
        # self.X = torch.load(f"x_{split}_pre.pt")
        self.subject_idxs = torch.load(os.path.join(data_dir, f"{split}_subject_idxs.pt"))

        if split in ["train", "val"]:
            self.y = torch.load(os.path.join(data_dir, f"{split}_y.pt"))
            assert len(torch.unique(self.y)) == self.num_classes, "Number of classes do not match."

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i):

        X =  self.X[i]
        sub_idx = self.subject_idxs[i]

        if hasattr(self, "y"): # selfがyを持っているか（train, valかtestかを判断）
            return X, self.y[i], sub_idx
        else:
            return X, sub_idx
        
    @property
    def num_channels(self) -> int:
        return self.X.shape[1]
    
    @property
    def seq_len(self) -> int:
        return self.X.shape[2]


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, image_files, preprocess):
        self.image_files = image_files
        self.preprocess = preprocess
        
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        image = self.preprocess(Image.open(image_path))
        # ファイルパスを分割
        parts = self.image_files[idx].split('\\')
        # 3つ目の部分を取得（0インデックスから数えるため）
        text = "This is picture of " + parts[-2]  # -2 は最後から2番目の部分
        return image, text


# yを画像埋込ベクトル（512）で返す
class ThingsMEGDatasetUseClip(torch.utils.data.Dataset):
    def __init__(self, split: str, data_dir: str = "data", clip_data=None) -> None:
        super().__init__()
        
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        self.split = split
        self.num_classes = 1854
        self.clip_data = clip_data
        
        self.X = torch.load(os.path.join(data_dir, f"{split}_X.pt"))
        self.X = pre_process(self.X, self.split)
        self.subject_idxs = torch.load(os.path.join(data_dir, f"{split}_subject_idxs.pt"))

        if split in ["train", "val"]:
            self.y = torch.load(os.path.join(data_dir, f"{split}_y.pt"))
            assert len(torch.unique(self.y)) == self.num_classes, "Number of classes do not match."

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i):

        X =  self.X[i]
        y = self.clip_data[self.y[i]]
        sub_idx = self.subject_idxs[i]

        if hasattr(self, "y"): # selfがyを持っているか（train, valかtestかを判断）
            return X, y, sub_idx
        else:
            return X, sub_idx
        
    @property
    def num_channels(self) -> int:
        return self.X.shape[1]
    
    @property
    def seq_len(self) -> int:
        return self.X.shape[2]