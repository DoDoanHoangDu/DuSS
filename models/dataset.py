from pathlib import Path
import random
from PIL import Image
import torch
from torchvision.transforms import v2
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self, hr_root=r"Data\train_hr", lr_root=r"Data\train_lr",
                 image_size=192, lr_suffix="x4", train=True, ratio=4):
        self.hr_root = Path(hr_root)
        self.lr_root = Path(lr_root)
        self.image_size = image_size
        self.lr_suffix = lr_suffix
        self.train = train
        self.ratio = ratio

        if image_size % ratio != 0:
            raise ValueError(f"image_size ({image_size}) must be divisible by ratio ({ratio})")

        self.lr_size = image_size // ratio
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

        self.image_paths = []
        for hr_path in self.hr_root.rglob("*"):
            if not hr_path.is_file() or hr_path.suffix.lower() not in extensions:
                continue

            lr_name = f"{hr_path.stem}{lr_suffix}{hr_path.suffix}"
            lr_path = self.lr_root / lr_name

            if not lr_path.is_file():
                raise RuntimeError(
                    f"LR image not found for HR image:\n"
                    f"  HR: {hr_path}\n"
                    f"  LR: {lr_path}"
                )

            self.image_paths.append((hr_path, lr_path))

        if not self.image_paths:
            raise RuntimeError(
                f"No paired images found under:\n"
                f"  HR: {self.hr_root}\n"
                f"  LR: {self.lr_root}"
            )

        self.to_tensor = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        hr_path, lr_path = self.image_paths[idx]

        with Image.open(hr_path) as image:
            hr = image.convert("RGB")
        with Image.open(lr_path) as image:
            lr = image.convert("RGB")

        if self.train:
            if hr.width < self.image_size or hr.height < self.image_size:
                raise ValueError(
                    f"HR image too small: {hr_path} "
                    f"({hr.width}x{hr.height}), "
                    f"required: {self.image_size}x{self.image_size}"
                )

            if lr.width < self.lr_size or lr.height < self.lr_size:
                raise ValueError(
                    f"LR image too small: {lr_path} "
                    f"({lr.width}x{lr.height}), "
                    f"required: {self.lr_size}x{self.lr_size}"
                )

            lr_x = random.randint(0, lr.width - self.lr_size)
            lr_y = random.randint(0, lr.height - self.lr_size)
            hr_x = lr_x * self.ratio
            hr_y = lr_y * self.ratio

            lr = lr.crop((lr_x, lr_y, lr_x + self.lr_size, lr_y + self.lr_size))
            hr = hr.crop((hr_x, hr_y, hr_x + self.image_size, hr_y + self.image_size))

            if random.random() > 0.5:
                angle = random.choice([
                    Image.Transpose.ROTATE_90,
                    Image.Transpose.ROTATE_180,
                    Image.Transpose.ROTATE_270,
                ])
                hr = hr.transpose(angle)
                lr = lr.transpose(angle)

            if random.random() < 0.5:
                hr = hr.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                lr = lr.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

            if random.random() < 0.5:
                hr = hr.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
                lr = lr.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

        hr = self.to_tensor(hr)
        lr = self.to_tensor(lr)

        return lr, hr