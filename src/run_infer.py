import sys
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com" 

import jittor as jt
from JDiffusion.pipelines import StableDiffusionPipeline
from peft import LoraConfig, get_peft_model
import json
import shutil
import argparse
import traceback

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(project_root, "src"))

from custom_pipeline import Img2ImgPipeline
from utils.config import load_config, set_seed
from utils.image import get_avg_image


def convert_diffusers_to_peft(diffusers_state_dict):
    """
    Turn Diffusers style dict (lora_linear_layer.up/down) 
    back into PEFT recognizable format (lora_A/B.default)
    """
    peft_dict = {}
    for k, v in diffusers_state_dict.items():
        # 1. Restore Up/Down mapping
        # Diffusers: .lora_linear_layer.down.weight -> PEFT: .lora_A.default.weight
        # Diffusers: .lora_linear_layer.up.weight   -> PEFT: .lora_B.default.weight
        new_k = k.replace("lora_linear_layer.down", "lora_A.default")
        new_k = new_k.replace("lora_linear_layer.up", "lora_B.default")
        
        # 2. Complete PEFT wrapping prefix (if your model is from get_peft_model)
        # if not new_k.startswith("base_model.model."):
        #     new_k = f"base_model.model.{new_k}"
            
        peft_dict[new_k] = v
    return peft_dict


def load_lora_robust(pipe, lora_path, te_lora_rk=4):
    print(f"[Custom Loader] Loading from: {lora_path}")
    weight_path = f"{lora_path}/pytorch_lora_weights.bin"
    if not os.path.exists(weight_path):
        print(f"Error: Weight file not found at {weight_path}")
        return

    state_dict = jt.load(weight_path)
    unet_keys = {}
    text_encoder_keys = {}

    for key, value in state_dict.items():
        if key.startswith("text_encoder"):
            # remove 'text_encoder.' prefix
            new_key = key.replace("text_encoder.", "")
            text_encoder_keys[new_key] = value
        elif key.startswith("unet"):
            # remove 'unet.' prefix
            new_key = key.replace("unet.", "")
            unet_keys[new_key] = value
        else:
            # fallback: keys without explicit prefix usually belong to UNet
            unet_keys[key] = value

    text_encoder_keys = convert_diffusers_to_peft(text_encoder_keys)
    
    print("   Cleaned Key Sample:")
    for sk in list(text_encoder_keys.keys())[:2]:
        print(f"      - {sk}")
    for sk in list(unet_keys.keys())[:2]:
        print(f"      - {sk}")

    # 1. Load UNet LoRA weights
    num_unet_keys = len(unet_keys)
    if num_unet_keys > 0:
        try:
            pipe.unet.load_attn_procs(unet_keys)
            print(f"[ok] UNet LoRA loaded ({num_unet_keys} keys)")
        except Exception as e:
            print(f"   x UNet Load Warning: {e}")
    
    # 2. Load Text Encoder LoRA weights
    num_te_keys = len(text_encoder_keys)
    if num_te_keys > 0 and te_lora_rk > 0:
        try:
            te_lora_config = LoraConfig(
                r=te_lora_rk, 
                lora_alpha=te_lora_rk,
                target_modules=["k_proj", "v_proj", "q_proj", "out_proj"]
            )
            pipe.text_encoder = get_peft_model(pipe.text_encoder, te_lora_config)
            pipe.text_encoder.load_state_dict(text_encoder_keys, strict=False)

            print(f"[ok] Text Encoder LoRA loaded ({num_te_keys} keys)")
        except TypeError as e:
            print(f"   x Text Encoder API Mismatch: {e}")
        except Exception as e:
            print(f"   x Text Encoder Load Failed: {e}")
            print("   (Ignored Text Encoder weights to allow inference continue)")
    else:
        print(f"   No Text Encoder weights found (config rank = {te_lora_rk})")
    
    return num_unet_keys > 0, num_te_keys > 0  # has_unet_lora, has_te_lora


def former_load_lora(pipe, lora_path):
    print(f"[Legacy Loader] Loading LoRA weights from: {lora_path}")
    try:
        pipe.load_lora_weights(lora_path)
        print("[ok] LoRA weights loaded successfully.")
    except Exception as e:
        print(f"   x LoRA Load Failed: {e}")
        print("-" * 30)
        print("(Full Traceback):")
        traceback.print_exc() 
        print("-" * 30)


def check_lora_existence(layer, name="Layer"):
    print(f"\n---  Probing LoRA for {name} ---")
    
    # 1. check attributes
    attributes = [attr for attr in dir(layer) if 'lora' in attr.lower()]
    print(f"   Attributes containing 'lora': {attributes}")

    # 2. check sub-modules (Jittor/PyTorch common)
    sub_modules = []
    if hasattr(layer, 'named_modules'):
        sub_modules = [n for n, m in layer.named_modules() if 'lora' in n.lower()]
        print(f"   Sub-modules containing 'lora': {sub_modules}")
    
    return len(attributes) > 0 and len(sub_modules) > 0


def run_single_task(taskid, yaml_config):
    dataset_root = os.path.join(project_root, "data", "train")
    outputs_root = os.path.join(project_root, "outputs")
    conf = load_config(yaml_config)
    
    print(f"\n[Worker] Processing Style {taskid} on GPU...")
    set_seed(conf.infer.seed + int(taskid))
    

    ### === [Step 1] Reload Pipeline (Clean State) ===
    print(f"Loading Custom Img2Img Pipeline...")
    pipe = Img2ImgPipeline.from_pretrained("Charles-Elena/stable-diffusion-2-1").to("cuda")

    # -- Debug: check TE weights before loading LoRA
    # unet_target = pipe.unet.down_blocks[0].attentions[0].transformer_blocks[0].attn1.to_q
    # te_target = pipe.text_encoder.text_model.encoder.layers[0].self_attn.q_proj
    # unet_lora_exists_before = check_lora_existence(unet_target, "[Before] UNet to_q")
    # te_lora_exists_before = check_lora_existence(te_target, "[Before] Text Encoder q_proj")


    ### === [Step 2] Robust Load LoRA ===
    lora_path = f"{outputs_root}/style_ckpt/style_{taskid}"
    has_unet_lora, has_te_lora = load_lora_robust(pipe, lora_path, conf.model.text_encoder_lora_rank)

    # -- Debug: check TE weights after loading LoRA
    # unet_target = pipe.unet.down_blocks[0].attentions[0].transformer_blocks[0].attn1.to_q
    # te_target = pipe.text_encoder.text_model.encoder.layers[0].self_attn.q_proj
    # unet_lora_exists_after = check_lora_existence(unet_target, "[After] UNet to_q")
    # te_lora_exists_after = check_lora_existence(te_target, "[After] Text Encoder q_proj")
    # if not unet_lora_exists_before and unet_lora_exists_after:
    #     print(" [ok] CONFIRMED: UNet LoRA weights injected in memory.")
    # if not te_lora_exists_before and te_lora_exists_after:
    #     print(" [ok] CONFIRMED: Text Encoder LoRA weights injected in memory.")
    
    print(f"\n--- Numerical Check Style {taskid} ---")
    # 1. Check UNet (sample Q projection layer)
    if (has_unet_lora):
        unet_weight = pipe.unet.down_blocks[0].attentions[0].transformer_blocks[0].attn1.to_q.lora_layer.up.weight
        unet_sum = unet_weight.abs().sum().item()
        print(f"   UNet Weight AbsSum: {unet_sum:.8f} ({'[ok] Loaded' if unet_sum > 0 else '[x] Zero!'})")
    # 2. Check Text Encoder (sample Q projection layer)
    if (has_te_lora):
        te_weight = pipe.text_encoder.base_model.model.text_model.encoder.layers[0].self_attn.q_proj.lora_B.default.weight
        te_sum = te_weight.abs().sum().item()
        print(f"   TE Weight AbsSum:   {te_sum:.8f} ({'[ok] Loaded' if te_sum > 0 else '[x] Zero!'})")


    ### === [Step 3] Prepare Average Image of training images ===
    train_img_folder = os.path.join(dataset_root, taskid, "images")
    avg_image_of_train = get_avg_image(train_img_folder, resolution=512)


    ### === [Step 4] Setup Directories ===
    save_dir = f"{outputs_root}/generate/{taskid}"
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok=True)


    ### === [Step 5] Read and Set Prompt ===
    prompt_file = os.path.join(dataset_root, taskid, "prompt.json")
    if not os.path.exists(prompt_file):
        print(f"Failed to process {taskid}: prompt.json not found.")
        return
    with open(prompt_file, "r") as file:
        prompts = json.load(file)
    # [fit] replace "_" to " " in each prompt
    prompt_items = [(id, prompt.replace("_", " ")) for id, prompt in prompts.items()]


    ### === [Step 6] Inference Generation ===
    batch_size = conf.infer.batch_size
    for i in range(0, len(prompt_items), batch_size):
        batch_items = prompt_items[i: i + batch_size]
        
        ## (1) Dealing with prompts
        # if conf.data.dynamic_prompt:
        #     input_prompt = f"an image of {prompt} in style_{taskid}"
        # else:
        #     input_prompt = prompt
        batch_origin_prompts = [item[1] for item in batch_items]
        batch_style_prompts = [f"an image of {p} in style_{taskid}" for p in batch_origin_prompts]
        print(f"  batch_style_prompts: {batch_style_prompts}")
        
        ## (2) batching other inputs
        batch_avg_images = [avg_image_of_train for _ in batch_items]
        neg_prompt = conf.infer.negative_prompt
        batch_neg_prompts = [neg_prompt for _ in batch_items] if neg_prompt else None

        print(f"   Batch {i//batch_size}: {len(batch_origin_prompts)} images")
        # image = pipe(input_prompt, num_inference_steps=25, width=512, height=512).images[0]
        results = pipe(
            # prompt fusion
            style_prompt=batch_style_prompts,
            origin_prompt=batch_origin_prompts,
            origin_scale=conf.infer.origin_scale,
            # avg image
            image=batch_avg_images,
            strength=conf.infer.strength,
            # other settings
            num_inference_steps=conf.infer.num_inference_steps,
            guidance_scale=conf.infer.guidance_scale,
            negative_prompt=batch_neg_prompts
        ).images

        for idx, img in enumerate(results):
            img_name = batch_origin_prompts[idx].replace(" ", "_") # [fit] replace " " back to "_"
            img.save(f"{save_dir}/{img_name}.png")

    del pipe
    jt.gc()
    print(f"[Worker] Style {taskid} Finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--taskid", type=str, required=True)
    # parser.add_argument("-e", "--exp_name", type=str, default=".")
    parser.add_argument("-c", "--config", type=str, default="self-tune.yaml")
    args = parser.parse_args()
    
    try:
        run_single_task(args.taskid, args.config)
    except Exception as e:
        print(f"[Error] Task {args.taskid} failed: {e}")
        traceback.print_exc()