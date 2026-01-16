import json, os, tqdm, shutil
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import jittor as jt
from JDiffusion.pipelines import StableDiffusionPipeline
from diffusers.loaders import LoraLoaderMixin
import argparse
import traceback
from peft import LoraConfig, get_peft_model


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


def load_lora_robust(pipe, lora_path):
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
    if num_te_keys > 0:
        try:
            te_lora_config = LoraConfig(
                r=16, 
                lora_alpha=16,
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
        print("   No Text Encoder weights found (Rank=0?)")


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num_styles", type=int, default=15)
    parser.add_argument("-e", "--exp_name", type=str, default=".")
    parser.add_argument("-s", "--seed", type=int, default=42, help="Seed for generation")
    args = parser.parse_args()

    max_num = args.num_styles
    dataset_root = "../data/train"
    outputs_root = f"../outputs/{args.exp_name}"

    # Plan B: Load LoRA in Pipeline : avoid unload_lora_weights not works so good
    print(f"=== Starting Inference for {max_num} styles ===")
    
    with jt.no_grad():
        for tempid in tqdm.tqdm(range(0, max_num)):
            taskid = "{:0>2d}".format(tempid)
            print(f"\nProcessing Style {taskid}...")
            
            ### === [Step 1] Reload Pipeline (Clean State) ===
            pipe = StableDiffusionPipeline.from_pretrained("Charles-Elena/stable-diffusion-2-1").to("cuda")

            # -- Debug: check TE weights before loading LoRA
            unet_target = pipe.unet.down_blocks[0].attentions[0].transformer_blocks[0].attn1.to_q
            te_target = pipe.text_encoder.text_model.encoder.layers[0].self_attn.q_proj
            unet_lora_exists_before = check_lora_existence(unet_target, "[Before] UNet to_q")
            te_lora_exists_before = check_lora_existence(te_target, "[Before] Text Encoder q_proj")
            
            ### === [Step 2] Robust Load LoRA ===
            lora_path = f"{outputs_root}/style_ckpt/style_{taskid}"
            load_lora_robust(pipe, lora_path)
            # former_load_lora(pipe, lora_path)

            # -- Debug: check TE weights after loading LoRA
            unet_target = pipe.unet.down_blocks[0].attentions[0].transformer_blocks[0].attn1.to_q
            te_target = pipe.text_encoder.text_model.encoder.layers[0].self_attn.q_proj
            unet_lora_exists_after = check_lora_existence(unet_target, "[After] UNet to_q")
            te_lora_exists_after = check_lora_existence(te_target, "[After] Text Encoder q_proj")
            if not unet_lora_exists_before and unet_lora_exists_after:
                print(" [ok] CONFIRMED: UNet LoRA weights injected in memory.")
            if not te_lora_exists_before and te_lora_exists_after:
                print(" [ok] CONFIRMED: Text Encoder LoRA weights injected in memory.")

            print(f"\n--- 💎 Numerical Check Style {taskid} ---")
            # 1. Check UNet (sample Q projection layer)
            unet_weight = pipe.unet.down_blocks[0].attentions[0].transformer_blocks[0].attn1.to_q.lora_layer.up.weight
            unet_sum = unet_weight.abs().sum().item()
            print(f"   UNet Weight AbsSum: {unet_sum:.8f} ({'✅ Loaded' if unet_sum > 0 else '❌ Zero!'})")
            
            # 2. Check Text Encoder (sample Q projection layer)
            te_weight = pipe.text_encoder.base_model.model.text_model.encoder.layers[0].self_attn.q_proj.lora_B.default.weight
            te_sum = te_weight.abs().sum().item()
            print(f"   TE Weight AbsSum:   {te_sum:.8f} ({'✅ Loaded' if te_sum > 0 else '❌ Zero!'})")

            ### === [Step 3] Setup Directories ===
            save_dir = f"{outputs_root}/generate/{taskid}"
            if os.path.exists(save_dir):
                shutil.rmtree(save_dir)
            os.makedirs(save_dir, exist_ok=True)

            ### === [Step 4] Generate ===
            prompt_file = f"{dataset_root}/{taskid}/prompt.json"
            if not os.path.exists(prompt_file):
                print(f"Skipping {taskid}: prompt.json not found.")
                continue

            with open(prompt_file, "r") as file:
                prompts = json.load(file)

            for id, prompt in prompts.items():
                jt.set_global_seed(args.seed)
                input_prompt = f"an image of {prompt} in style_{taskid}"
                print(f"   Generating: {input_prompt}") 
                image = pipe(input_prompt, num_inference_steps=25, width=512, height=512).images[0]
                image.save(f"{save_dir}/{prompt}.png")
            
            del pipe
            jt.gc()


if __name__ == "__main__":
    main()
    

# parser = argparse.ArgumentParser()
# parser.add_argument("-n", "--num_styles", type=int, default=15)
# parser.add_argument("-e", "--exp_name", type=str, default=".")
# parser.add_argument("-s", "--seed", type=int, default=42, help="Seed for generation")
# args = parser.parse_args()

# max_num = args.num_styles
# dataset_root = "../data/train"
# outputs_root = f"../outputs/{args.exp_name}"

# pipe = StableDiffusionPipeline.from_pretrained("Charles-Elena/stable-diffusion-2-1").to("cuda")


# with jt.no_grad():
#     for tempid in tqdm.tqdm(range(0, max_num)):
#         taskid = "{:0>2d}".format(tempid)
#         # pipe = StableDiffusionPipeline.from_pretrained("Charles-Elena/stable-diffusion-2-1").to("cuda")  # plan B
#         pipe.load_lora_weights(f"{outputs_root}/style_ckpt/style_{taskid}")
        
#         save_dir = f"{outputs_root}/generate/{taskid}"
#         if os.path.exists(save_dir):
#             shutil.rmtree(save_dir)
#         os.makedirs(save_dir, exist_ok=True)

#         # load json
#         with open(f"{dataset_root}/{taskid}/prompt.json", "r") as file:
#             prompts = json.load(file)

#         for id, prompt in prompts.items():
#             jt.set_global_seed(args.seed)
#             print(prompt)
#             input_prompt = f"an image of {prompt} in style_{taskid}"
#             image = pipe(input_prompt, num_inference_steps=25, width=512, height=512).images[0]
#             image.save(f"{save_dir}/{prompt}.png")

#         try:
#             pipe.unload_lora_weights()
#         except AttributeError:
#             print("warning: unload_lora_weights not found, cannot unload lora weights. Should use plan B")