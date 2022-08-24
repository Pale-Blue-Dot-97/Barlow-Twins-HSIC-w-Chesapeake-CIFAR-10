import os
from typing import Any, Callable, List, Optional, Tuple
from glob import glob
from collections import Counter

import numpy as np
from torchvision.datasets import VisionDataset
from PIL import Image
import rasterio as rt
from alive_progress import alive_bar


class Chesapeake_CIFAR10(VisionDataset):
    
    image_glob = "*_RGBI.tif"
    targets_glob = "*_MASK.tif"
    
    def __init__(
        self, 
        root: str,
        train: bool = True,
         
        transforms: Optional[Callable] = None, 
        transform: Optional[Callable] = None, 
        target_transform: Optional[Callable] = None
        ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        
        self.train = train
        
        sub_dir = "train" if self.train else "test"
        sub_dir_path = os.path.join(self.root, sub_dir)
        
        image_filenames = glob(os.path.join(sub_dir_path, self.image_glob))
        target_filenames = glob(os.path.join(sub_dir_path, self.targets_glob))
        
        self.data = np.empty((len(image_filenames), 4, 32, 32))
        self.targets = np.empty((len(image_filenames), 4, 32, 32))
        
        print("Constructing Image Dataset")
        for i, filename in enumerate(image_filenames):
            with rt.open(filename) as src:
                self.data[i] = src.read()

        print("Constructing Target Dataset")
        for i, filename in enumerate(target_filenames):
            with rt.open(filename) as src:
                self.targets[i] = src.read()
        
        self.classes = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        label = Counter(target).most_common()[0][0]
        return img, label

    def __len__(self) -> int:
        return len(self.data)