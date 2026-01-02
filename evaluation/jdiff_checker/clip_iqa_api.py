from PIL import Image
import numpy as np
from torchmetrics.multimodal import CLIPImageQualityAssessment
import torch
import os
from torchvision.transforms import Compose, Resize, ToTensor, Lambda
# _ = torch.manual_seed(42)
# imgs = torch.randint(255, (2, 3, 224, 224)).float()
# metric = CLIPImageQualityAssessment()
# metric(imgs)
'''
  * quality: "Good photo." vs "Bad photo."
        * brightness: "Bright photo." vs "Dark photo."
        * noisiness: "Clean photo." vs "Noisy photo."
        * colorfullness: "Colorful photo." vs "Dull photo."
        * sharpness: "Sharp photo." vs "Blurry photo."
        * contrast: "High contrast photo." vs "Low contrast photo."
        * complexity: "Complex photo." vs "Simple photo."
        * natural: "Natural photo." vs "Synthetic photo."
        * happy: "Happy photo." vs "Sad photo."
        * scary: "Scary photo." vs "Peaceful photo."
        * new: "New photo." vs "Old photo."
        * warm: "Warm photo." vs "Cold photo."
        * real: "Real photo." vs "Abstract photo."
        * beautiful: "Beautiful photo." vs "Ugly photo."
        * lonely: "Lonely photo." vs "Sociable photo."
        * relaxing: "Relaxing photo." vs "Stressful photo."'''
transform = Compose([
    # Resize((299, 299)),  # 调整图像大小以匹配Inception模型的输入尺寸
    ToTensor(),  # 将PIL图像转换为Tensor
    Lambda(lambda x: (x * 255))  # 归一化到0-255并转为uint8
])
metric = CLIPImageQualityAssessment(prompts=("quality",
                                             #  "brightness", "noisiness", "colorfullness", "sharpness", "contrast", "complexity", "natural", "happy", "scary", "new", "warm", "real", "beautiful", "lonely", "relaxing"
                                             ))
metric = metric.to("cuda")


def get_clip_iqa(images):
    # images = torch.stack(images)
    # gt_images = torch.stack(gt_images)
    images = [transform(img) for img in images]
    # gt_images = transform(gt_images)
    images = torch.stack(images).to("cuda")

    score = metric(images)
    # breakpoint()
    return score.cpu().numpy().tolist()
    # result = {}
    # for key, value in score.items():
    #     result[key] = value.cpu().numpy().tolist()
    # result['summary'] = {}
    # for key, value in result.items():
    #     if key == "summary":
    #         continue
        # result['summary'][key] = np.mean(value)
    return result


if __name__ == '__main__':
    style_number = 42
    result = {}
    for idx in range(style_number):
        idx = str(idx)
        # gt_path = f"/home/ldy/ldy/jtcomp/gt/{idx}/images"
        gen_path = f"/home/ldy/ldy/jtcomp/output_rename/{idx}/"
        # gt_imgs = os.listdir(gt_path)
        gen_imgs = os.listdir(gen_path)
        # gt_imgs = [os.path.join(gt_path, img) for img in gt_imgs]
        gen_imgs = [os.path.join(gen_path, img) for img in gen_imgs]
        # gt_imgs = [Image.open(img) for img in gt_imgs]
        gen_imgs = [Image.open(img) for img in gen_imgs]
        # fid_score = get_fid_score(gen_imgs, gt_imgs)
        # print(fid_score)
        # result[idx] = fid_score.item()
        result[idx] = clip_iqa(gen_imgs)
        # print(result[idx])
        # exit()
    import json
    with open("clip_iqa_result.json", "w") as f:
        json.dump(result, f)
