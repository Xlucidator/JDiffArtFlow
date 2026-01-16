import os
import jittor as jt
from jittor.compatibility.optim import AdamW
from diffusers.optimization import get_scheduler
from diffusers.loaders import LoraLoaderMixin
from diffusers.utils import convert_state_dict_to_diffusers
from peft.utils import get_peft_model_state_dict
from tqdm.auto import tqdm
import math


class BaseTrainer:

    def __init__(self, config, engine, dataloader):
        self.config = config
        self.engine = engine
        self.dataloader = dataloader
        self.project_dir = config.experiment.output_dir
        os.makedirs(self.project_dir, exist_ok=True)

        # overrode_max_train_steps = False
        # num_update_steps_per_epoch = math.ceil(len(dataloader) / config.train.gradient_accumulation_steps)
        # if config.train.max_train_steps is None:
        #     config.train.max_train_steps = config.train.num_train_epochs * num_update_steps_per_epoch
        #     overrode_max_train_steps = True

        self.optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.engine.unet.parameters()),
            lr=config.train.learning_rate,
            betas=(config.train.adam_beta1, config.train.adam_beta2),
            weight_decay=config.train.adam_weight_decay,
            eps=config.train.adam_epsilon,
        )
        # self.optimizer = AdamW(
        #     filter(lambda p: p.requires_grad, self.engine.unet.parameters()),
        #     lr=float(config.train.learning_rate),
        #     betas=(float(config.train.adam_beta1), float(config.train.adam_beta2)),
        #     weight_decay=float(config.train.adam_weight_decay),
        #     eps=float(config.train.adam_epsilon),
        # )

        self.lr_scheduler = get_scheduler(
            config.train.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=config.train.lr_warmup_steps,
            num_training_steps=config.train.max_train_steps,
        )


    def train(self):
        print(f"***** Starting Training: {self.config.experiment.name} *****")
        global_step = 0
        progress_bar = tqdm(range(self.config.train.max_train_steps), desc="Steps")

        epoch = 0
        while global_step < self.config.train.max_train_steps:
            for batch in self.dataloader:
                loss = self.compute_loss(batch)
                loss.backward()

                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()

                progress_bar.update(1)
                global_step += 1
                progress_bar.set_postfix(loss=loss.detach().item())

                if global_step >= self.config.train.max_train_steps:
                    break
            epoch += 1
            
        self.save_checkpoint()


    def compute_loss(self, batch):
        raise NotImplementedError
    
    
    def save_checkpoint(self):
        print(f"Saving LoRA weights to {self.project_dir}")
        unet = self.engine.unet.to(jt.float32)
        unet_lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
        LoraLoaderMixin.save_lora_weights(
            save_directory=self.project_dir,
            unet_lora_layers=unet_lora_state_dict,
            safe_serialization=False
        )