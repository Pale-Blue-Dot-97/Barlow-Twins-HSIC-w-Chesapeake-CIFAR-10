import os
from typing import Any, Callable, Optional, Tuple
from collections import Counter

import torch
from torchvision.datasets import VisionDataset
from PIL import Image


class Chesapeake_CIFAR10(VisionDataset):
    
    image_fn = "RGBI.pt"
    targets_fn = "MASK.pt"
    
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
        
        image_path = os.path.join(sub_dir_path, self.image_fn)
        target_path = os.path.join(sub_dir_path, self.targets_fn)
        
        self.data = torch.load(image_path).type(torch.FloatTensor)
        self.targets = torch.load(target_path).type(torch.FloatTensor)
        
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
        #img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        print(f"{target=}")
        modes = Counter(target).most_common()
        
        print(f"{modes=}")
        
        label = modes[0][0]
        return img, label

    def __len__(self) -> int:
        return len(self.data)
