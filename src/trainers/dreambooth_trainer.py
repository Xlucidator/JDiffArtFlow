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
        # (Option) Mixed with Prior Preservation Loss
        if self.config.data.with_prior_preservation:
            # batch = [instance_images, class_images]
            model_pred_inst, model_pred_prior = model_pred.chunk(2, dim=0)
            target_inst, target_prior = target.chunk(2, dim=0)
            loss_inst  = nn.mse_loss(model_pred_inst, target_inst)
            loss_prior = nn.mse_loss(model_pred_prior, target_prior)
            # mixed loss
            loss = loss_inst + self.config.data.prior_loss_weight * loss_prior
        else:
            loss = nn.mse_loss(model_pred, target)
        
        return loss