from pathlib import Path
from PIL import Image
from tqdm import tqdm

def downscale_images(source_folder="Data/train_hr", target_folder="Data/train_lr", ratio=4):
    source_folder = Path(source_folder)
    target_folder = Path(target_folder)
    target_folder.mkdir(parents=True, exist_ok=True)
    extensions = {".jpg",".jpeg",".png",".bmp",".webp",}

    image_paths = [path for path in source_folder.iterdir() if path.is_file() and path.suffix.lower() in extensions]
    for image_path in tqdm(image_paths):
        with Image.open(image_path) as img:
            width, height = img.size
            lr_size = (width // ratio, height // ratio)
            lr = img.resize(lr_size, Image.BICUBIC)
            output_path = target_folder / image_path.name
            lr.save(output_path)


if __name__ == "__main__":
    downscale_images("Data/train_hr", "Data/train_lr")
    downscale_images("Data/valid_hr", "Data/valid_lr")