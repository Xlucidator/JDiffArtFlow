
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import json
import jittor as jt
from JDiffusion import StableDiffusionPipeline, UniPCMultistepScheduler
import argparse
import tqdm

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
train_data_dir = os.path.join(project_root, "data", "train")
prior_output_dir = os.path.join(project_root, "data", "prior")

def get_train_prompts(images_dir):
    """Get prompt from image filenames in the given directory."""
    if not os.path.exists(images_dir):
        return []
    
    img_extensions = {'.png', '.jpg', '.jpeg'}
    prompts = set()
    for filename in os.listdir(images_dir):
        name, ext = os.path.splitext(filename)
        if ext.lower() in img_extensions:
            # boat.png -> boat; olive_tree.png -> olive tree
            prompt = name.replace("_", " ").lower()
            prompts.add(prompt)
    return list(prompts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-np", "--num_images_per_prompt", type=int, default=8, help="pictures per prompt image")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--include_infer", action='store_true', help="whether to include prompts from prompt.json")
    args = parser.parse_args()

    # 1. Load Model
    print("Loading SD 2.1 Model for Prior Generation...")
    pipe = StableDiffusionPipeline.from_pretrained("Charles-Elena/stable-diffusion-2-1").to("cuda")
    pipe.scheduler = UniPCMultistepScheduler().from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)

    # 2. Traverse Each Style
    if not os.path.exists(train_data_dir):
        print(f"Error: Data root {train_data_dir} does not exist.")
        return

    tasks = sorted(os.listdir(train_data_dir))
    
    with jt.no_grad():
        for taskid in tqdm.tqdm(tasks, desc="Tasks"):
            task_path = os.path.join(train_data_dir, taskid)
            if not os.path.isdir(task_path): continue

            # === A. Get Prompt ===
            # 1. from train data (如 boat.png -> "boat")
            train_images_dir = os.path.join(task_path, "images")
            prompts = get_train_prompts(train_images_dir)
            
            # 2. (option) from prompt.json get inference prompts
            # try to prevent OOD during inference
            if args.include_infer:
                json_path = os.path.join(task_path, "prompt.json")
                if os.path.exists(json_path):
                    try:
                        with open(json_path, 'r') as f:
                            infer_prompts = json.load(f).values()
                            prompts.extend([p.lower() for p in infer_prompts])
                    except:
                        pass

            prompts = sorted(list(set(prompts)))
            
            if not prompts:
                print(f"Skipping {taskid}: No prompts found.")
                continue

            # === B. Prepare Save Directory ===
            # data/prior/00/
            save_parent = os.path.join(prior_output_dir, taskid)
            os.makedirs(save_parent, exist_ok=True)

            # === C. Generate Images ===
            print(f"  Generating priors for Style {taskid}: {prompts}")
            for prompt in prompts:
                input_prompt = f"A {prompt} in the center"
                for i in range(args.num_images_per_prompt):
                    # Check if already exists (avoid rerun)
                    filename = f"{prompt.replace(' ', '_')}_{i}.png"
                    save_path = os.path.join(save_parent, filename)
                    
                    if os.path.exists(save_path):
                        continue
                        
                    image = pipe(input_prompt, num_inference_steps=25, width=512, height=512).images[0]
                    image.save(save_path) # data/prior/00/boat_0.png

    print("Prior generation finished.")

if __name__ == "__main__":
    main()