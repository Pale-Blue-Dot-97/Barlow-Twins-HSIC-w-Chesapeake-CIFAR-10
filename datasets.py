import os
from typing import Any, Callable, List, Optional, Tuple
from glob import glob
from torchvision.datasets import VisionDataset
from PIL import Image


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
        
        self.data: List[Any] = []
        self.targets: List[Any] = []
        
        sub_dir = "train" if self.train else "test"
        
        sub_dir_path = os.path.join(self.root, sub_dir)
        for filename in glob(self.image_glob, root_dir=sub_dir_path):
            pass
        
    
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

        return img, target

    def __len__(self) -> int:
        return len(self.data)