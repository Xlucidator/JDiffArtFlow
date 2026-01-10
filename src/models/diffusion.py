import jittor as jt
from transformers import AutoTokenizer, PretrainedConfig
from JDiffusion import AutoencoderKL, UNet2DConditionModel
from diffusers import DDPMScheduler
from peft import LoraConfig


class DiffusionEngine:
    ''' load all diffusion components and manage lora '''
    
    def __init__(self, config):
        self.config = config
        model_id = config.model.pretrained_model_name_or_path
        
        # load model components
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, subfolder="tokenizer", use_fast=False)
        self.noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
        self.vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
        self.unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
        self.text_encoder = self._load_text_encoder(model_id)

        # freeze parameters and setup lora
        self._freeze_params()
        self._setup_lora()

    
    def _load_text_encoder(self, model_id):
        ''' load text encoder based on architecture '''
        text_encoder_config = PretrainedConfig.from_pretrained(model_id, subfolder="text_encoder")
        model_class = text_encoder_config.architectures[0]
        if model_class == "CLIPTextModel":
            from transformers import CLIPTextModel
            return CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
        elif model_class == "T5EncoderModel":
            from transformers import T5EncoderModel
            return T5EncoderModel.from_pretrained(model_id, subfolder="text_encoder")
        else:
            raise ValueError(f"{model_class} not supported.")
        
    
    def _freeze_params(self):
        ''' freeze all model parameters '''
        for param in self.vae.parameters():
            param.requires_grad = False
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        for param in self.unet.parameters():
            param.requires_grad = False
        # TODO useless


    def _setup_lora(self):
        unet_lora_config = LoraConfig(
            r=self.config.model.rank,
            lora_alpha=self.config.model.rank,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0", "add_k_proj", "add_v_proj"],
        )
        self.unet.add_adapter(unet_lora_config)

        for name, param in self.unet.named_parameters():
             if "lora" in name:
                 param.requires_grad = True
