from typing import Any, Callable, Dict, List, Optional, Union
import PIL
import jittor as jt
from JDiffusion.pipelines import StableDiffusionPipeline
from JDiffusion.pipelines.pipeline_stable_diffusion_jittor import retrieve_timesteps, rescale_noise_cfg
from diffusers.image_processor import PipelineImageInput
from JDiffusion.pipelines.pipeline_output_jittor import StableDiffusionPipelineOutput
from diffusers.utils import deprecate
from JDiffusion.utils import randn_tensor


def retrieve_latents(encoder_output: jt.Var, seed: int = None, sample_mode: str = "sample"):
    ''' safely retrieve latents from vae encoder output '''
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(seed)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")
    

class Img2ImgPipeline(StableDiffusionPipeline):
    ''' Stable Diffusion Img2Img Pipeline with Jittor backend '''

    def prepare_latents(
        self, image, timestep, batch_size, num_images_per_prompt, dtype, 
        seed=None, add_noise=True
    ):
        ''' Generate latents from the input image '''
        if not isinstance(image, (jt.Var, PIL.Image.Image, list)):
            raise ValueError(f"`image` has to be of type `jittor.Tensor`, `PIL.Image.Image` or list but is {type(image)}")

        image = image.to(dtype=dtype)
        batch_size = batch_size * num_images_per_prompt

        ### 1. Encode the image into latents
        if image.shape[1] == 4:
            ## assume latents are already provided
            init_latents = image
        else:
            ## vae encoding logic
            # (option) cast type to float32 to avoid overflow
            if self.vae.config.force_upcast:
                image = image.float()
                self.vae.to(dtype=jt.float32)

            # deal with multiple seeds
            if isinstance(seed, list):
                if len(seed) != batch_size:
                    raise ValueError(f"Seed list length {len(seed)} does not match batch size {batch_size}.")
                # Manual batch encoding for different seeds
                init_latents = [
                    retrieve_latents(self.vae.encode(image[i: i+1]), seed=seed[i])
                    for i in range(batch_size)
                ]
                init_latents = jt.concat(init_latents, dim=0)
            else:
                init_latents = retrieve_latents(self.vae.encode(image), seed=seed)

            # (option) cast type back
            if self.vae.config.force_upcast:
                self.vae.to(dtype)
            
            # standard scaling
            init_latents = init_latents.to(dtype)
            init_latents = self.vae.config.scaling_factor * init_latents

        ### 2. Adjust batch size of latents
        current_batch_size = init_latents.shape[0]
        if batch_size > current_batch_size:
            if batch_size % current_batch_size != 0:
                raise ValueError(f"Cannot duplicate image of size {current_batch_size} to {batch_size}.")
            repeat_times = batch_size // current_batch_size
            init_latents = init_latents.repeat(repeat_times, 1, 1, 1)

        ### 3. Add noise (Option)
        if add_noise:
            shape = init_latents.shape
            noise = randn_tensor(shape, seed=seed,  dtype=dtype) # keep random generator consistent
            init_latents = self.scheduler.add_noise(init_latents, noise, timestep)

        return init_latents


    def get_timesteps(self, num_inference_steps, strength, denoising_start=None):
        ### === Case A: Standard Img2Img (controlled by Strength) ===
        if denoising_start is None:
            ## calculate skip steps based on strength
            init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
            t_start = max(num_inference_steps - init_timestep, 0)

            ## direct slicing, skip steps = order * t_start
            timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :] 
            return timesteps, num_inference_steps - t_start

        ### === Case B: Advanced control (precise control via denoising_start) ===
        else:
            ## calculate absolute cutoff timestep (e.g., 800)
            discrete_timestep_cutoff = int(
                round(
                    self.scheduler.config.num_train_timesteps  # num_train_timesteps is usually 1000
                    - (denoising_start * self.scheduler.config.num_train_timesteps)
                )
            )
            timesteps = self.scheduler.timesteps  # get all timesteps
            num_inference_steps = (timesteps < discrete_timestep_cutoff).sum().item() # steps less than cutoff
            if self.scheduler.order == 2 and num_inference_steps % 2 == 0: 
                num_inference_steps = num_inference_steps + 1 # special case for 2nd order schedulers

            # slice from the end
            timesteps = timesteps[-num_inference_steps:]
            return timesteps, num_inference_steps


    @jt.no_grad()
    def __call__(
        self,
        style_prompt: Union[str, List[str]] = None,
        origin_prompt: Union[str, List[str]] = None,
        origin_scale: float = 0.0,  # fix type
        image: PipelineImageInput = None,
        strength: float = 0.3,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        seed: Optional[Union[int, List[int]]] = None,
        latents: Optional[jt.Var] = None,
        prompt_embeds: Optional[jt.Var] = None,
        negative_prompt_embeds: Optional[jt.Var] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        **kwargs,
    ):

        # 1. Setup
        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs
        self._interrupt = False

        # 2. Get batch size
        if style_prompt is not None and isinstance(style_prompt, str):
            batch_size = 1
        elif style_prompt is not None and isinstance(style_prompt, list):
            batch_size = len(style_prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # 3. Encode input prompt
        lora_scale = (
            self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
        )
        # optimize for single prompt case
        if origin_scale == 0.0:
            # pure style, compile style_prompt
            prompt_embeds, negative_prompt_embeds = self.encode_prompt(
                style_prompt, num_images_per_prompt, self.do_classifier_free_guidance,
                negative_prompt, prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds,
                lora_scale=lora_scale, clip_skip=self.clip_skip,
            )
        elif origin_scale == 1.0:
            # pure origin, compile origin_prompt
            prompt_embeds, negative_prompt_embeds = self.encode_prompt(
                origin_prompt, num_images_per_prompt, self.do_classifier_free_guidance,
                negative_prompt, prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds,
                lora_scale=lora_scale, clip_skip=self.clip_skip,
            )
        else:
            # need to blend, encode twice and weight
            style_embeds, style_neg_embeds = self.encode_prompt(
                style_prompt, num_images_per_prompt, self.do_classifier_free_guidance,
                negative_prompt, prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds,
                lora_scale=lora_scale, clip_skip=self.clip_skip,
            )
            origin_embeds, origin_neg_embeds = self.encode_prompt(
                origin_prompt, num_images_per_prompt, self.do_classifier_free_guidance,
                negative_prompt, prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds,
                lora_scale=lora_scale, clip_skip=self.clip_skip,
            )
            # Interpolation
            prompt_embeds = style_embeds * (1 - origin_scale) + origin_embeds * origin_scale
            negative_prompt_embeds = style_neg_embeds * (1 - origin_scale) + origin_neg_embeds * origin_scale
        
        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if self.do_classifier_free_guidance:
            prompt_embeds = jt.concat([negative_prompt_embeds, prompt_embeds])

        # 4. Preprocess image
        image = self.image_processor.preprocess(image)

        # 5. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, timesteps)
        timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength, denoising_start=None)
        latent_timestep = jt.Var(timesteps[:1]).long().repeat(batch_size * num_images_per_prompt)

        # 6. Prepare latent variables
        latents = self.prepare_latents(
            image, latent_timestep, batch_size, num_images_per_prompt,
            prompt_embeds.dtype, seed=seed, add_noise=True, # add_noise=True
        )
    
        # 6.2 Guidance Scale Embedding (Optional)
        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = jt.array(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(dtype=latents.dtype)

        # 7. Denoising loop (remove warmup steps)
        extra_step_kwargs = self.prepare_extra_step_kwargs(eta)
        self._num_timesteps = len(timesteps)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt: continue

                latent_model_input = jt.concat([latents] * 2) if self.do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                noise_pred = self.unet(
                    latent_model_input, t, encoder_hidden_states=prompt_embeds,
                    timestep_cond=timestep_cond, cross_attention_kwargs=self.cross_attention_kwargs,
                    return_dict=False,
                )[0]

                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=self.guidance_rescale)

                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {k: locals()[k] for k in callback_on_step_end_tensor_inputs}
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                progress_bar.update()

        if not output_type == "latent":
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
            image, has_nsfw_concept = self.run_safety_checker(image, prompt_embeds.dtype)
        else:
            image = latents
            has_nsfw_concept = None

        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)
