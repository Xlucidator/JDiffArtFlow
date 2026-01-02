from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to("cuda")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
import numpy as np
# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)
@torch.no_grad()
def clip_r_api(image,text):
    inputs = processor(text=text, images=image, return_tensors="pt", padding=True)
    inputs = inputs.to("cuda")
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image # this is the image-text similarity score
    probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities
    #get top k labels in text
    k = 5
    result = []
    topk = torch.topk(probs, k)
    topk_result = topk.indices[0]
    for i in range(k):
        result.append((text[topk_result[i].item()],topk.values[0][i].item()))
    return result


    # breakpoint()
# print(probs)
    # logits_per_image = outputs.logits_per_image # this is the image-text similarity score
    # breakpoint()
    # score = logits_per_image/100
    # score = ((logits_per_image/10)**2)/10
    # score = score.exp()/np.exp(1)
    # return score.item()

# image = Image.open("/home/ldy/ldy/jtcomp/reff.png")
# text = "a camera"
# print(clip_text_alignment(image,text))
# text = "a dog"
# print(clip_text_alignment(image,text))
# inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True)

# outputs = model(**inputs)
# logits_per_image = outputs.logits_per_image # this is the image-text similarity score
# # probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities
# print(probs)
import os
import json
import tqdm
if __name__=='__main__':
    result = {}
    with open("/home/ldy/ldy/jtcomp/prompt.text", "r") as f:
        prompts = f.readlines()
    prompts = [prompt.strip().lower() for prompt in prompts]
    # prompts = set(prompts)
    # print(prompts)
    images_path ='/home/ldy/ldy/jtcomp/style-aligned/sta_output'
    total = 0
    correct = 0
    for root,dirs,files in os.walk(images_path):
        for dir in dirs:
            result[dir] = {}
            images_path = os.path.join(root,dir)
            images = os.listdir(images_path)
            for image_name in tqdm.tqdm(images):
                image_path = os.path.join(images_path,image_name)
                image_prompt = image_name.split(".")[0].lower()
                image = Image.open(image_path)
                input_prompts = list(set(prompts+[image_prompt]))
                image_result = clip_text_alignment(image,prompts)
                correct_ = False
                for i in range(5):
                    if image_result[i][0] == image_prompt:
                        correct_ = True
                        break
                if correct_:
                    correct += 1
                total += 1
                result[dir][image_name] = {"correct":correct_,"result":image_result}
    result['ratio'] = correct/total
    with open("/home/ldy/ldy/jtcomp/clip_r_result_sta.json","w") as f:
        json.dump(result,f)
    print(correct/total)