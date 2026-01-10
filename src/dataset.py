import jittor as jt
from jittor import transform
from jittor.compatibility.utils.data import Dataset
from pathlib import Path
from PIL import Image
from PIL.ImageOps import exif_transpose
from transformers import AutoTokenizer


class DreamBoothDataset(Dataset):
    def __init__(self, config, tokenizer):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer

        self.instance_data_root = Path(config.instance_data_dir)
        if not self.instance_data_root.exists():
            raise ValueError("Instance images root doesn't exists.")
        
        self.instance_images_path = list(self.instance_data_root.iterdir())
        self.num_instance_images = len(self.instance_images_path)
        self.instance_prompt = config.instance_prompt
        self._length = self.num_instance_images

        self.image_transforms = transform.Compose([
            transform.Resize(config.resolution),
            transform.CenterCrop(config.resolution) if config.center_crop else transform.RandomCrop(config.resolution),
            transform.ToTensor(),
            transform.ImageNormalize([0.5], [0.5]),
        ])

    def __len__(self):
        return self._length
    
    def __getitem__(self, index):
        example = {}
        instance_image = Image.open(self.instance_images_path[index % self.num_instance_images])
        instance_image = exif_transpose(instance_image)

        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        example["instance_images"] = self.image_transforms(instance_image)

        # TODO