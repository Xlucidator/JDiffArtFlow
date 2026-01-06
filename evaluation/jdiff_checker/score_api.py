from lpips_api import lpips_score_api
from fid_api import fid_score_api
from hist_api import hist_score_api
import argparse
from dino_api import dino_style_similarity
# from clip_api import clip_text_alignment
from clip_r_api import clip_r_api
import os
import json
from PIL import Image
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def get_score(upload_path, result_path):
    # ref_images = {}

    ref_style_numbers = 15
    A_ref_gt_path = "../../data/A_gt"
    # A_ref_gt_prompts_path = os.path.join(A_ref_gt_path, "prompts.json")
    # load prompts
    # with open(A_ref_gt_prompts_path, 'r') as f:
    #     A_ref_gt_prompts = json.load(f)
    # load style ref images
    gt_style_img_paths = {}
    upload_style_img_path = {}
    for style_idx in range(ref_style_numbers):
        style_idx_str = str(style_idx).zfill(2)
        style_idx_str = str(style_idx).zfill(2)
        gt_style_path = os.path.join(A_ref_gt_path, style_idx_str, "images")
        upload_style_path = os.path.join(upload_path, style_idx_str)
        gt_style_img_paths[style_idx_str] = gt_style_path
        upload_style_img_path[style_idx_str] = upload_style_path

    # clip_scores_summary, clip_scores = get_clip_scores(upload_style_img_path)
    dino_scores_summary, dino_scores = get_dino_scores(
        upload_style_img_path, gt_style_img_paths)
    hist_scores_summary, hist_scores = get_hist_scores(
        upload_style_img_path, gt_style_img_paths)
    fid_score = get_fid_scores(upload_style_img_path, gt_style_img_paths)
    lpip_score = get_lpips_score(upload_style_img_path, gt_style_img_paths)
    clip_iqa_score_summary,clip_iqa_score = get_clip_iqa_score(upload_style_img_path)
    clip_r_score, clip_r_scores = get_clip_r_score(upload_style_img_path)
    # breakpoint()
    # ref_images[style_idx_str] = []
    # for img in os.listdir(style_path):
    #     ref_images[style_idx_str].append(
    #         (img, Image.open(os.path.join(style_path, img))))

    # load upload images
    # upload_images = {}
    # for upload_style_idx in range(ref_style_numbers):
    #     # upload_style_idx_str = str(upload_style_idx).zfill(3)
    #     upload_style_idx_str = str(upload_style_idx)
    #     upload_images[upload_style_idx_str] = []
    #     upload_style_path = os.path.join(upload_path, upload_style_idx_str)
    #     for img in os.listdir(upload_style_path):
    #         upload_images[upload_style_idx_str].append(
    #             (img, Image.open(os.path.join(upload_style_path, img))))
    # get clip scores
    # clip_scores_summary, clip_scores = get_clip_scores(upload_images)

    # # get dino scores
    # dino_scores_summary, dino_scores = get_dino_scores(
    #     upload_images, ref_images)
    # # save result
    result = {
        # "clip_summary": clip_scores_summary,
          "dino_summary": dino_scores_summary,
        # "clip_scores": clip_scores, 
        "dino_scores": dino_scores,
        "hist_summary": hist_scores_summary, "hist_scores": hist_scores,
        'fid_score': fid_score, 'lpip_score': lpip_score,'clip_iqa_summary':clip_iqa_score_summary,'clip_iqa_score':clip_iqa_score,
        # "clip_score": clip_scores_summary['total'], 
        "dino_score": dino_scores_summary['total'],
        'hist_score': hist_scores_summary['total'], 'fid_score': fid_score,'lpip_score': lpip_score['total'],
        'clip_iqa_score':clip_iqa_score_summary['total'],
        'clip_r_score':clip_r_score,'clip_r_scores':clip_r_scores
    }
    
    os.makedirs(result_path, exist_ok=True)
    with open(os.path.join(result_path, "result.json"), 'w') as f:
        json.dump(result, f, indent=4, cls=NumpyEncoder, sort_keys=True)

    return {
        # "clip_score": clip_scores_summary['total'],
        "dino_score": dino_scores_summary['total'],
        'hist_score': hist_scores_summary['total'],
        'fid_score': fid_score,
        'lpip_score': lpip_score['total'],
        'clip_iqa_score':clip_iqa_score_summary['total'],
        'clip_r_score':clip_r_score
    }

def get_clip_r_score(upload_img_paths):
    # breakpoint()
    result = {}
    with open("../clip_r_prompts.txt", "r") as f:
        prompts = f.readlines()
    prompts = [prompt.strip().lower() for prompt in prompts]
    # prompts = set(prompts)
    # print(prompts)
    images_path =upload_img_paths
    total = 0
    correct = 0
    # for root,dirs,files in os.walk(images_path):
        # for dir in dirs:
    for dir in upload_img_paths.keys():
            result[dir] = {}
            images_path = upload_img_paths[dir]
            images = os.listdir(images_path)
            for image_name in images:
                image_path = os.path.join(images_path,image_name)
                image_prompt = image_name.split(".")[0].lower()
                image = Image.open(image_path)
                input_prompts = list(set(prompts+[image_prompt]))
                image_result = clip_r_api(image,input_prompts)
                correct_ = False
                for i in range(5):
                    if image_result[i][0] == image_prompt:
                        correct_ = True
                        break
                if correct_:
                    correct += 1
                total += 1
                result[dir][image_name] = {"correct":correct_,"result":image_result}
    result['total_correct_ratio'] = correct/total
    return result['total_correct_ratio'],result

'''
def get_clip_scores(upload_img_paths):
    clip_scores = {}
    for style_idx_str in upload_img_paths.keys():
        clip_scores[style_idx_str] = []
        img_paths = os.listdir(upload_img_paths[style_idx_str])
        img_paths = [os.path.join(
            upload_img_paths[style_idx_str], img) for img in img_paths]
        for img_path in img_paths:
            # breakpoint()
            img = Image.open(img_path)
            img_name = os.path.basename(img_path)

            prompt = img_name.split(".")[0]
            clip_scores[style_idx_str].append(
                (img_name, clip_text_alignment(image=img, text=prompt)))
    clip_scores_summary = {}
    clip_scores_list = []
    for style_idx_str in clip_scores.keys():
        style_idx_score = np.mean(
            [score for img_name, score in clip_scores[style_idx_str]])
        clip_scores_summary[style_idx_str] = style_idx_score
        clip_scores_list.append(style_idx_score)
    clip_scores_summary['total'] = np.mean(clip_scores_list)
    return clip_scores_summary, clip_scores

'''
def get_dino_scores(upload_img_paths, gt_img_paths):
    dino_scores = {}
    for style_idx_str in upload_img_paths.keys():
        up_dino_scores = {}
        # breakpoint()
        img_paths = os.listdir(upload_img_paths[style_idx_str])
        img_paths = [os.path.join(
            upload_img_paths[style_idx_str], img) for img in img_paths]
        for upload_img_path in img_paths:
            up_img_name = os.path.basename(upload_img_path)
            up_img = Image.open(upload_img_path)
            up_dino_scores[up_img_name] = []
            gtgt_img_paths = os.listdir(gt_img_paths[style_idx_str])
            gtgt_img_paths = [os.path.join(
                gt_img_paths[style_idx_str], img) for img in gtgt_img_paths]
            for gt_img_path in gtgt_img_paths:
                gt_img_name = os.path.basename(gt_img_path)
                gt_img = Image.open(gt_img_path)
                up_dino_scores[up_img_name].append(
                    (gt_img_name, dino_style_similarity(gt_img, up_img)))
        dino_scores[style_idx_str] = up_dino_scores

    dino_scores_summary = {}
    for style_idx_str in dino_scores.keys():
        dino_scores_summary_item = {}
        for up_img_name in dino_scores[style_idx_str].keys():
            up_img_style_score = np.mean(
                [score for gt_img_name, score in dino_scores[style_idx_str][up_img_name]])
            dino_scores_summary_item[up_img_name] = up_img_style_score

        dino_scores_summary_item['total'] = np.mean(
            [score for score in dino_scores_summary_item.values()])
        dino_scores_summary[style_idx_str] = dino_scores_summary_item

    dino_scores_summary['total'] = np.mean(
        [score['total'] for score in dino_scores_summary.values()])
    return dino_scores_summary, dino_scores


def get_hist_scores(upload_img_paths, gt_img_paths):
    hist_scores = {}
    for style_idx_str in upload_img_paths.keys():
        up_imgs = os.listdir(upload_img_paths[style_idx_str])
        gt_imgs = os.listdir(gt_img_paths[style_idx_str])
        up_imgs = [os.path.join(upload_img_paths[style_idx_str], img) for img in up_imgs]
        gt_imgs = [os.path.join(gt_img_paths[style_idx_str], img) for img in gt_imgs]
        hist_score = hist_score_api(
            up_imgs, gt_imgs)
        hist_scores[style_idx_str] = hist_score
    hist_scores_summary = {}
    hist_scores_summary['total'] = np.mean(
        [score for score in hist_scores.values()])
    return hist_scores_summary, hist_scores


def get_fid_scores(upload_img_paths, gt_img_paths):
    upload_file_paths = {}
    gt_file_paths = {}
    for style_idx_str in upload_img_paths.keys():
        upload_file_paths[style_idx_str] = os.listdir(
            upload_img_paths[style_idx_str])
        gt_file_paths[style_idx_str] = os.listdir(gt_img_paths[style_idx_str])
        upload_file_paths[style_idx_str] = [os.path.join(
            upload_img_paths[style_idx_str], img) for img in upload_file_paths[style_idx_str]]
        gt_file_paths[style_idx_str] = [os.path.join(
            gt_img_paths[style_idx_str], img) for img in gt_file_paths[style_idx_str]]
    # print(upload_file_paths, gt_file_paths)
    fid_score = fid_score_api(upload_file_paths, gt_file_paths)
    return fid_score


def get_lpips_score(upload_img_paths, gt_img_paths):
    lpip_scores = {}
    for style_idx_str in upload_img_paths.keys():
        up_imgs = os.listdir(upload_img_paths[style_idx_str])
        gt_imgs = os.listdir(gt_img_paths[style_idx_str])
        up_imgs = [os.path.join(upload_img_paths[style_idx_str], img)
                   for img in up_imgs]
        gt_imgs = [os.path.join(gt_img_paths[style_idx_str], img)
                   for img in gt_imgs]
        up_imgs = [Image.open(img) for img in up_imgs]
        gt_imgs = [Image.open(img) for img in gt_imgs]
        lpips_score = lpips_score_api(up_imgs, gt_imgs)
        lpip_scores[style_idx_str] = lpips_score

    lpip_scores['total'] = np.mean([score for score in lpip_scores.values()])
    return lpip_scores
from clip_iqa_api import get_clip_iqa
def get_clip_iqa_score(upload_img_paths):
    clip_iqa_scores = {}
    for style_idx_str in upload_img_paths.keys():
        img_paths = os.listdir(upload_img_paths[style_idx_str])
        img_paths = [os.path.join(
            upload_img_paths[style_idx_str], img) for img in img_paths]
        img_paths = [Image.open(img) for img in img_paths]
        clip_iqa_score = get_clip_iqa(img_paths)
        clip_iqa_scores[style_idx_str] = clip_iqa_score
    clip_iqa_summary = {}
    for style_idx_str in clip_iqa_scores.keys():
        clip_iqa_summary[style_idx_str] = np.mean(
            clip_iqa_scores[style_idx_str])
    clip_iqa_summary['total'] = np.mean([score for score in clip_iqa_summary.values()])
    # clip_iqa_scores['total'] = np.mean([score for score in clip_iqa_scores.values()])
    return clip_iqa_summary, clip_iqa_scores


def print_in_format(result):
    print("\n" + "="*50)
    print(f"{'METRIC NAME':<25} | {'SCORE':<15}")
    print("-" * 50)

    for key, value in result.items():
        val_float = float(value) if isinstance(value, (np.floating, float)) else value
        print(f"{key:<25} | {val_float:.6f}")
    
    print("="*50 + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # upload path result path
    parser.add_argument('--upload_path', type=str)
    parser.add_argument('--result_path', type=str)
    args = parser.parse_args()
    upload_path = args.upload_path
    result_path = args.result_path

    import time
    start_time = time.time()
    res = get_score(upload_path, result_path)
    print_in_format(res)
    print("time: ", time.time() - start_time)
    # print(get_clip_scores(upload_images))
