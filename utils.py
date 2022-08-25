from PIL import Image
import torch
from torch import Tensor
from torchvision import transforms
from torchvision.transforms import functional_tensor as ft


class DetachedColorJitter(transforms.ColorJitter):
    """Sends RGB channels of multi-spectral images to be transformed by :class:`ColorJitter`."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, img: Tensor) -> Tensor:
        """Detaches RGB channels of input image to be sent to :class:`ColorJitter`.
        All other channels bypass :class:`ColorJitter` and are concatenated onto the colour jittered RGB channels.
        Args:
            img (Tensor): Input image.
        Raises:
            ValueError: If number of channels of input ``img`` is 2.
        Returns:
            Tensor: Color jittered image.
        """
        channels = ft.get_image_num_channels(img)

        jitter_img : Tensor
        if channels > 3:
            rgb_jitter = super().forward(img[:3])
            jitter_img = torch.cat((rgb_jitter, img[3:]), 0)

        elif channels in (1, 3):
            jitter_img = super().forward(img)

        else:
            raise ValueError(f"{channels} channel images are not supported!")

        return jitter_img


class ChesapeakeCifarPairTransform:
    def __init__(self, train_transform = True, pair_transform = True):
        if train_transform is True:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(32),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply([DetachedColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                #transforms.RandomGrayscale(p=0.2),
                #transforms.ToTensor(),
                # Note that 4th band values are copied from Red band.
                transforms.Normalize([0.4914, 0.4822, 0.4465, 0.4914], [0.2023, 0.1994, 0.2010, 0.2023])])
        else:
            self.transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize([0.4914, 0.4822, 0.4465, 0.4914], [0.2023, 0.1994, 0.2010, 0.2023])])
        self.pair_transform = pair_transform
    def __call__(self, x):
        if self.pair_transform is True:
            y1 = self.transform(x)
            y2 = self.transform(x)
            return y1, y2
        else:
            return self.transform(x)


# for cifar10 (32x32)
class CifarPairTransform:
    def __init__(self, train_transform = True, pair_transform = True):
        if train_transform is True:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(32),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])
        else:
            self.transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])
        self.pair_transform = pair_transform
    def __call__(self, x):
        if self.pair_transform is True:
            y1 = self.transform(x)
            y2 = self.transform(x)
            return y1, y2
        else:
            return self.transform(x)

# for tiny imagenet (64x64)
class TinyImageNetPairTransform:
    def __init__(self, train_transform = True, pair_transform = True):
        if train_transform is True:
            self.transform = transforms.Compose([
                    transforms.RandomApply(
                        [transforms.ColorJitter(brightness=0.4, contrast=0.4, 
                                                saturation=0.4, hue=0.1)], 
                        p=0.8
                    ),
                    transforms.RandomGrayscale(p=0.1),
                    transforms.RandomResizedCrop(
                        64,
                        scale=(0.2, 1.0),
                        ratio=(0.75, (4 / 3)),
                        interpolation=Image.BICUBIC,
                    ),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ToTensor(),
                    transforms.Normalize((0.480, 0.448, 0.398), (0.277, 0.269, 0.282))
                ])
        else:
            self.transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.480, 0.448, 0.398), (0.277, 0.269, 0.282))
                ])
        self.pair_transform = pair_transform
    def __call__(self, x):
        if self.pair_transform is True:
            y1 = self.transform(x)
            y2 = self.transform(x)
            return y1, y2
        else:
            return self.transform(x)

# for stl10 (96x96)
class StlPairTransform:
    def __init__(self, train_transform = True, pair_transform = True):
        if train_transform is True:
            self.transform = transforms.Compose([
                    transforms.RandomApply(
                        [transforms.ColorJitter(brightness=0.4, contrast=0.4, 
                                                saturation=0.4, hue=0.1)], 
                        p=0.8
                    ),
                    transforms.RandomGrayscale(p=0.1),
                    transforms.RandomResizedCrop(
                        64,
                        scale=(0.2, 1.0),
                        ratio=(0.75, (4 / 3)),
                        interpolation=Image.BICUBIC,
                    ),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ToTensor(),
                    transforms.Normalize((0.43, 0.42, 0.39), (0.27, 0.26, 0.27))
                ])
        else:
            self.transform = transforms.Compose([
                    transforms.Resize(70, interpolation=Image.BICUBIC),
                    transforms.CenterCrop(64),
                    transforms.ToTensor(),
                    transforms.Normalize((0.43, 0.42, 0.39), (0.27, 0.26, 0.27))
                ])
        self.pair_transform = pair_transform
    def __call__(self, x):
        if self.pair_transform is True:
            y1 = self.transform(x)
            y2 = self.transform(x)
            return y1, y2
        else:
            return self.transform(x)
