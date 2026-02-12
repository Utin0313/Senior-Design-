from PIL import Image # type: ignore
from pathlib import Path
import numpy as np # type: ignore

def add_black_noise_to_image(img: Image.Image, noise_ratio=0.05) -> Image.Image:
    arr = np.array(img.convert("RGB"))

    h, w, c = arr.shape
    total_pixels = h * w
    num_noisy = int(total_pixels * noise_ratio)

    ys = np.random.randint(0, h, num_noisy)
    xs = np.random.randint(0, w, num_noisy)

    arr[ys, xs] = [0, 0, 0]

    return Image.fromarray(arr)


def process_folder(input_folder: Path, output_folder: Path, noise_ratio=0.05):
    output_folder.mkdir(parents=True, exist_ok=True)

    for img_path in input_folder.glob("*"):
        if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp", ".tif"]:
            continue  # skip non-images

        img = Image.open(img_path)
        noisy_img = add_black_noise_to_image(img, noise_ratio)

        output_path = output_folder / img_path.name
        noisy_img.save(output_path)

        print(f"Saved noisy image: {output_path}")


# -------- Run here ---------

input_folder = Path.home() / "Downloads" / "ENGIN 491 (SD)" / "images"
output_folder = Path.home() / "Downloads" / "ENGIN 491 (SD)" / "noisy_images"

process_folder(input_folder, output_folder, noise_ratio=0.05)

