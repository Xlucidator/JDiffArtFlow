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
        self.encoder_hidden_states = None
        self.class_prompt_encoder_hidden_states = None
        self.tokenizer_max_length = config.tokenizer_max_length

        self.instance_data_root = Path(config.instance_data_dir)
        if not self.instance_data_root.exists():
            raise ValueError(f"Instance images root doesn't exists: {self.instance_data_root}")
        
        self.instance_images_path = list(self.instance_data_root.iterdir())
        self.num_instance_images = len(self.instance_images_path)
        self.instance_prompt = config.instance_prompt
        self._length = self.num_instance_images

        # Dealing with prior preservation class images
        if config.with_prior_preservation and config.class_data_dir is not None:
            self.class_data_root = Path(config.class_data_dir)
            self.class_data_root.mkdir(parents=True, exist_ok=True)
            self.class_images_path = list(self.class_data_root.iterdir())
            self.num_class_images = min(len(self.class_images_path), config.num_class_images)
            self._length = max(self.num_class_images, self.num_instance_images)
            self.class_prompt = config.class_prompt
        else:
            self.class_data_root = None

        self.image_transforms = transform.Compose([
            transform.Resize(config.resolution),
            transform.CenterCrop(config.resolution) if config.center_crop else transform.RandomCrop(config.resolution),
            transform.ToTensor(),
            transform.ImageNormalize([0.5], [0.5]),
        ])

        print(f"  Num examples = {len(self)}")


    def __len__(self):
        return self._length
    

    def __getitem__(self, index):
        example = {}

        ## Dealing with instance Images and Prompts
        image_path = self.instance_images_path[index % self.num_instance_images]
        # 1. packing image
        instance_image = Image.open(image_path)
        instance_image = exif_transpose(instance_image)
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        example["instance_images"] = self.image_transforms(instance_image)
        # 2. packing prompt
        if self.config.dynamic_prompt:
            object_name = image_path.stem.replace("_", " ")
            dynamic_prompt = f"an image of {object_name} in {self.instance_prompt}"
            text_inputs = self._tokenize_prompt(dynamic_prompt, self.tokenizer_max_length)
        else:
            text_inputs = self._tokenize_prompt(self.instance_prompt, self.tokenizer_max_length)
        example["instance_prompt_ids"] = text_inputs.input_ids
        example["instance_attention_mask"] = text_inputs.attention_mask

        ## Dealing with prior preservation class Images and Prompts
        if self.class_data_root:
            class_image = Image.open(self.class_images_path[index % self.num_class_images])
            class_image = exif_transpose(class_image)
            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            example["class_images"] = self.image_transforms(class_image)
            class_text_inputs = self._tokenize_prompt(self.class_prompt, self.tokenizer_max_length)
            example["class_prompt_ids"] = class_text_inputs.input_ids
            example["class_attention_mask"] = class_text_inputs.attention_mask

        return example
    

    def _tokenize_prompt(self, prompt, tokenizer_max_length=None):
        if tokenizer_max_length is not None:
            max_length = tokenizer_max_length
        else:
            max_length = self.tokenizer.model_max_length
        
        text_inputs = self.tokenizer(
            prompt,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        return text_inputs


def collate_fn(examples):
    has_attention_mask = "instance_attention_mask" in examples[0]

    input_ids = [example["instance_prompt_ids"] for example in examples]
    pixel_values = [example["instance_images"] for example in examples]
    if has_attention_mask:
        attention_mask = [example["instance_attention_mask"] for example in examples]

    pixel_values = jt.stack(pixel_values).float()
    input_ids = jt.cat(input_ids, dim=0)

    batch = {"input_ids": input_ids, "pixel_values": pixel_values}
    if has_attention_mask:
        attention_mask = jt.cat(attention_mask, dim=0)
        batch["attention_mask"] = attention_mask

    return batch