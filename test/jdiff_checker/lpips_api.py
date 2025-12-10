import torch
# _ = torch.manual_seed(123)
from torchvision.transforms import Compose, Resize, ToTensor, Lambda
from torchmetrics.functional.image.lpips import learned_perceptual_image_patch_similarity
# img1 = (torch.rand(10, 3, 100, 100) * 2) - 1
# img2 = (torch.rand(10, 3, 100, 100) * 2) - 1
# score = learned_perceptual_image_patch_similarity(img1, img2, net_type='squeeze')
# # print(score)
transform = Compose([
    Resize((256,256)),  # 调整图像大小以匹配Inception模型的输入尺寸
    ToTensor(),  # 将PIL图像转换为Tensor
    Lambda(lambda x: (x * 2)-1)  # 归一化到0-255并转为uint8
])
def lpips_score_api(images, gt_images):
    images = [transform(img) for img in images]
    gt_images = [transform(img) for img in gt_images]
    input_images = []
    gt_input_images = []
    for img in images:
        for gt_img in gt_images:
            input_images.append(img)
            gt_input_images.append(gt_img)
    images = torch.stack(input_images).to("cuda")
    gt_images = torch.stack(gt_input_images).to("cuda")
    #C = 3
    images = images[:, :3, :, :]
    gt_images = gt_images[:, :3, :, :]
    # breakpoint()
    #
    score = learned_perceptual_image_patch_similarity(images, gt_images, net_type='squeeze')
    return score.item()
import os
from PIL import Image
import tqdm
if __name__=='__main__':
    style_number = 42
    result = {}
    for idx in tqdm.tqdm( range(style_number)):
        idx = str(idx)
        gt_path = f"/home/ldy/ldy/jtcomp/gt/{idx}/images"
        gen_path = f"/home/ldy/ldy/jtcomp/output_rename/{idx}/"
        # gen_path = gt_path
        gt_imgs = os.listdir(gt_path)
        gen_imgs = os.listdir(gen_path)
        gt_imgs = [os.path.join(gt_path, img) for img in gt_imgs]
        gen_imgs = [os.path.join(gen_path, img) for img in gen_imgs]
        gt_imgs = [Image.open(img) for img in gt_imgs]
        gen_imgs = [Image.open(img) for img in gen_imgs]
        fid_score = lpips_api(gen_imgs, gt_imgs)
        # print(fid_score)
        result[idx] = fid_score
    import json
    with open("lpips_result.json", "w") as f:
        json.dump(result, f)