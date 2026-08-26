from pathlib import Path
import cv2
from tqdm import tqdm

from pathlib import Path
import cv2
from tqdm import tqdm


def extract_paired_patches(
    hr_folder,
    lr_folder,
    output_hr_folder,
    output_lr_folder,
    patch_size=256,
    stride=256,
    lr_suffix="x4",
    scale=4,
):
    hr_folder = Path(hr_folder)
    lr_folder = Path(lr_folder)
    output_hr_folder = Path(output_hr_folder)
    output_lr_folder = Path(output_lr_folder)

    output_hr_folder.mkdir(parents=True, exist_ok=True)
    output_lr_folder.mkdir(parents=True, exist_ok=True)

    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    lr_patch_size = patch_size // scale
    count = 0

    hr_images = [
        p for p in hr_folder.rglob("*")
        if p.is_file() and p.suffix.lower() in extensions
    ]

    for hr_path in tqdm(hr_images):
        lr_name = f"{hr_path.stem}{lr_suffix}{hr_path.suffix}"
        lr_path = lr_folder / lr_name

        if not lr_path.exists():
            print(f"Skipping, LR not found: {lr_path}")
            continue

        hr = cv2.imread(str(hr_path), cv2.IMREAD_COLOR)
        lr = cv2.imread(str(lr_path), cv2.IMREAD_COLOR)

        if hr is None or lr is None:
            print(f"Skipping unreadable pair: {hr_path}")
            continue

        hr_h, hr_w = hr.shape[:2]
        lr_h, lr_w = lr.shape[:2]

        if hr_w != lr_w * scale or hr_h != lr_h * scale:
            print(
                f"Skipping mismatched dimensions: "
                f"{hr_path.name} HR={hr_w}x{hr_h}, "
                f"{lr_path.name} LR={lr_w}x{lr_h}"
            )
            continue

        if hr_h < patch_size or hr_w < patch_size:
            continue

        # Generate positions and always include the boundary
        y_positions = list(range(0, hr_h - patch_size + 1, stride))
        x_positions = list(range(0, hr_w - patch_size + 1, stride))

        last_y = hr_h - patch_size
        last_x = hr_w - patch_size

        if y_positions[-1] != last_y:
            y_positions.append(last_y)

        if x_positions[-1] != last_x:
            x_positions.append(last_x)

        for y in y_positions:
            for x in x_positions:
                hr_patch = hr[y:y + patch_size, x:x + patch_size]

                lr_x = x // scale
                lr_y = y // scale

                lr_patch = lr[
                    lr_y:lr_y + lr_patch_size,
                    lr_x:lr_x + lr_patch_size
                ]

                if lr_patch.shape[:2] != (lr_patch_size, lr_patch_size):
                    continue

                filename = f"{count:08d}.png"

                cv2.imwrite(
                    str(output_hr_folder / filename),
                    hr_patch
                )
                cv2.imwrite(
                    str(output_lr_folder / filename),
                    lr_patch
                )

                count += 1

    print(f"Extracted {count} paired patches.")

extract_paired_patches(
    hr_folder="Data/DIV2K_valid_HR",
    lr_folder="Data/DIV2K_valid_LR_bicubic/X4",
    output_hr_folder="Data/valid_HR",
    output_lr_folder="Data/valid_LR",
    patch_size=256 * 4,
    stride=256 * 4,
    lr_suffix="x4",
    scale=4
)