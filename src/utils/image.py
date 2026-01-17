import os
from PIL import Image
import numpy as np


def get_avg_image(style_folder, resolution=512):
    """
    Calculate the average image from all images in the given folder.
    - If the folder does not exist or contains no images, return a black image.
    """
    if not os.path.exists(style_folder):
        print(f"Warning: {style_folder} not found, using black image.")
        return Image.new('RGB', (resolution, resolution), (0, 0, 0))

    img_list = os.listdir(style_folder)
    img_list = [f for f in img_list if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not img_list:
        return Image.new('RGB', (resolution, resolution), (0, 0, 0))

    images = []
    for img_name in img_list:
        img_path = os.path.join(style_folder, img_name)
        try:
            img = Image.open(img_path).convert("RGB").resize((resolution, resolution))
            images.append(np.array(img))
        except Exception as e:
            print(f"Error reading {img_path}: {e}")

    avg_data = np.mean(images, axis=0).astype(np.uint8)
    return Image.fromarray(avg_data)