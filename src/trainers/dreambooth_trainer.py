import jittor as jt
import jittor.nn as nn
from .base_trainer import BaseTrainer

class DreamBoothTrainer(BaseTrainer):
    def compute_loss(self, batch):
        ### Model Inputs
        pixel_values = batch["pixel_values"]
        input_ids = batch["input_ids"]
        
        ### Model Forward
        # 1. VAE encoding -> Latents
        latents = self.engine.vae.encode(pixel_values).latent_dist.sample()
        latents = latents * self.engine.vae.config.scaling_factor
        # 2. Add noise with random noise and timesteps
        noise = jt.randn_like(latents)
        bsz = latents.shape[0]
        timesteps = jt.randint(0, self.engine.noise_scheduler.config.num_train_timesteps, (bsz,)).long()
        noisy_latents = self.engine.noise_scheduler.add_noise(latents, noise, timesteps)
        # 3. Text encoding
        encoder_hidden_states = self.engine.text_encoder(input_ids)[0]
        # 4. UNet prediction
        model_pred = self.engine.unet(noisy_latents, timesteps, encoder_hidden_states).sample
        # 5. Handle prediction type (epsilon vs v_prediction)
        if self.engine.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.engine.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.engine.noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError("Unknown prediction type")

        ### Get Final Loss
        loss = nn.mse_loss(model_pred, target)
        return loss