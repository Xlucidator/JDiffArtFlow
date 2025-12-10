from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to("cuda")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
import numpy as np
# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)
@torch.no_grad()
def clip_text_alignment(image,text):
    inputs = processor(text=text, images=image, return_tensors="pt", padding=True)
    inputs = inputs.to("cuda")
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image # this is the image-text similarity score
    # breakpoint()
    score = logits_per_image/100
    # score = ((logits_per_image/10)**2)/10
    # score = score.exp()/np.exp(1)
    return score.item()

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