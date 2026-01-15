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
            raise ValueError(f"Instance images root doesn't exists: {self.instance_data_root}")
        
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

        # Tokenize prompts
        text_inputs = self.tokenizer(
            self.instance_prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        example["instance_prompt_ids"] = text_inputs.input_ids
        example["instance_attention_mask"] = text_inputs.attention_mask

        return example


def collate_fn(examples):
    input_ids = [example["instance_prompt_ids"] for example in examples]
    pixel_values = [example["instance_images"] for example in examples]
    attention_mask = [example["instance_attention_mask"] for example in examples]

    pixel_values = jt.stack(pixel_values).float()
    input_ids = jt.cat(input_ids, dim=0)

    batch = {"input_ids": input_ids, "pixel_values": pixel_values}
    if attention_mask:
        batch["attention_mask"] = attention_mask

    return batch