import numpy as np
from PIL import Image
import os
import torch
from torchvision.transforms import Compose, Resize, ToTensor, Lambda
# _ = torch.manual_seed(123)
from torchmetrics.image.fid import FrechetInceptionDistance
transform = Compose([
    # Resize((299, 299)),  # 调整图像大小以匹配Inception模型的输入尺寸
    ToTensor(),  # 将PIL图像转换为Tensor
    Lambda(lambda x: (x * 255).byte())  # 归一化到0-255并转为uint8
])


def get_fid_score(images, gt_images):
    # images = [Image.open(img) for img in images]
    # gt_images = [Image.open(img) for img in gt_images]
    # images: PIL Image list
    # gt_images: PIL Image list
    images = [transform(img) for img in images]
    gt_images = [transform(img) for img in gt_images]
    images = torch.stack(images).to("cuda")
    gt_images = torch.stack(gt_images).to("cuda")
    # print(images.shape)
    # NHWC to NCHW
    # breakpoint()
    # images = images.permute(0, 3, 1, 2)
    # gt_images = gt_images.permute(0, 3, 1, 2)
    fid = FrechetInceptionDistance(feature=64).to("cuda")
    fid.update(gt_images, real=True)
    fid.update(images, real=False)
    return fid.compute().item()





def fid_score_api(gen_img_paths, gt_img_paths):
    gt_imgs_total = []
    gen_imgs_total = []
    for style_idx_str in gen_img_paths.keys():
        gen_imgs = gen_img_paths[style_idx_str]
        gt_imgs = gt_img_paths[style_idx_str]
        gen_imgs = [Image.open(img) for img in gen_imgs]
        gt_imgs = [Image.open(img) for img in gt_imgs]
    gt_imgs_total.extend(gt_imgs)
    gen_imgs_total.extend(gen_imgs)
    fid_score = get_fid_score(gen_imgs_total, gt_imgs_total)
    return fid_score


# fid = FrechetInceptionDistance(feature=64)
# # generate two slightly overlapping image intensity distributions
# imgs_dist1 = torch.randint(0, 200, (100, 3, 299, 299), dtype=torch.uint8)
# imgs_dist2 = torch.randint(100, 255, (100, 3, 299, 299), dtype=torch.uint8)
# fid.update(imgs_dist1, real=True)
# fid.update(imgs_dist2, real=False)
# fid.compute()
# def fid_score_api(gen_img_paths, ref_img_paths):


#     return avg_score
if __name__ == '__main__':
    style_number = 28
    result = {}
    gt_imgs_total = []
    gen_imgs_total = []
    for idx in range(style_number):
        idx = str(idx).zfill(2)
        # gt_path = f"/home/ldy/ldy/jtcomp/gt_output/{idx}/"
        gt_path = f"/home/ldy/ldy/jtcomp/jdiff2024/gt_data/B_gt/{idx}/images/"
        gen_path = f"/home/ldy/ldy/jtcomp/jdiff2024/upload/101623/2024-08-15/2014721_22-37-42/{idx}/"
        # gen_path = f"/home/ldy/ldy/jtcomp/output43/{idx}/"
        gt_imgs = os.listdir(gt_path)
        gen_imgs = os.listdir(gen_path)
        gt_imgs = [os.path.join(gt_path, img) for img in gt_imgs]
        gen_imgs = [os.path.join(gen_path, img) for img in gen_imgs]
        gt_imgs = [Image.open(img) for img in gt_imgs]
        gen_imgs = [Image.open(img) for img in gen_imgs]
        gt_imgs_total.extend(gt_imgs)
        gen_imgs_total.extend(gen_imgs)
    fid_score = get_fid_score(gen_imgs_total, gt_imgs_total)
    # fid_score = get_fid_score(gen_imgs, gt_imgs)
    # print(fid_score)
    # result[idx] = fid_score.item()
    result['fid'] = fid_score
    print(result)
    # import json
    # with open("fid_result.json", "w") as f:
    #     json.dump(result, f)
