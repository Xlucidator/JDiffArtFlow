import json, os, tqdm, shutil
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import jittor as jt
from JDiffusion.pipelines import StableDiffusionPipeline
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--num_styles", type=int, default=15)
args = parser.parse_args()

max_num = args.num_styles
dataset_root = "../data/train"
outputs_root = "../outputs"

pipe = StableDiffusionPipeline.from_pretrained("Charles-Elena/stable-diffusion-2-1").to("cuda")


with jt.no_grad():
    for tempid in tqdm.tqdm(range(0, max_num)):
        taskid = "{:0>2d}".format(tempid)
        # pipe = StableDiffusionPipeline.from_pretrained("Charles-Elena/stable-diffusion-2-1").to("cuda")  # plan B
        pipe.load_lora_weights(f"{outputs_root}/style_ckpt/style_{taskid}")
        
        save_dir = f"{outputs_root}/generate/{taskid}"
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        os.makedirs(save_dir, exist_ok=True)

        # load json
        with open(f"{dataset_root}/{taskid}/prompt.json", "r") as file:
            prompts = json.load(file)

        for id, prompt in prompts.items():
            print(prompt)
            image = pipe(prompt + f" in style_{taskid}", num_inference_steps=25, width=512, height=512).images[0]
            image.save(f"{save_dir}/{prompt}.png")

        try:
            pipe.unload_lora_weights()
        except AttributeError:
            print("warning: unload_lora_weights not found, cannot unload lora weights. Should use plan B")