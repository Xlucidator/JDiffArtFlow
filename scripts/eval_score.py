import json
import argparse
import os
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
outputs_path = os.path.join(project_root, "outputs")

def compute_final_score(metrics):
    """
    compute_final_score with the formula: (Style Similarity) × (Text Consistency + Image Quality)
    args:
        - metrics (dict): containing 'dino_score', 'lpip_score', 'clip_r_score', 'fid_score', 'clip_iqa_score'
    returns:
        - final_score (float): the final score formula:
          [DinoV2 * 0.2 + Pixel-Hist * 0.4 + (1-LPIPS) * 0.4] * [CLIP-R * 0.5 + (20 - min(FID,20)) / 20 * 0.4 + CLIP-IQA * 0.1]
        - components  (dict): calculated values of each component (Style, Text, Quality)
    """
    
    ## === 1. Metrics Extract ===
    # high-is-better
    dino = metrics.get('dino_score', 0.0) 
    clip_r = metrics.get('clip_r_score', 0.0)
    clip_iqa = metrics.get('clip_iqa_score', 0.0)
    hist = metrics.get('hist_score', 0.0)
    
    # low-is-better
    lpips = metrics.get('lpip_score', 1.0)      # 0~1 
    fid = metrics.get('fid_score', 100.0)       # >0
    

    ## === 2. Calculate Components ===
    # --- A. Style Similarity ---
    # Determined by LPIPS and DinoV2.
    # - Dino : focuses on "semantic/structural layout", 
    # - LPIPS: focuses on "perceptual/texture details", distance metric -> convert to similarity (1 - LPIPS)
    term_dino = dino * 0.2
    term_hist = hist * 0.4
    term_lpip = max(0, 1 - lpips) * 0.4
    style_component = term_dino + term_hist + term_lpip

    # --- B. Text Consistency ---
    # Directly use CLIP-R (Recall)
    term_clip_r = clip_r * 0.5
    text_component = term_clip_r
    
    # --- C. Images Quality ---
    # Determined by CLIP-IQA and FID.
    # - FID: distance metric, lower is better, need normalization
    # - CLIP-IQA: 0~1 aesthetic quality score
    term_fid = (20 - min(fid, 20)) / 20 * 0.4
    term_clip_iqa = clip_iqa * 0.1
    quality_component = term_fid + term_clip_iqa


    ## === 3. Final Calculation ===
    # Formula: (Style Similarity) × (Text Consistency + Image Quality)
    final_score = style_component * (text_component + quality_component)
    
    return final_score, {
        "style_component": style_component,
        "text_component": text_component,
        "quality_component": quality_component,
    }


def print_report(final_score, components):
    print("\n" + "="*40)
    print("            Score Report ")
    print("="*40)
    print(f"1. Style Component  : {components['style_component']:.4f}")
    print(f"2. Text Component   : {components['text_component']:.4f}")
    print(f"3. Quality Component: {components['quality_component']:.4f}")
    print("-" * 40)
    print(f"  Final Score: {final_score:.6f}")
    print("="*40 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default=".")
    args = parser.parse_args()

    score_path = os.path.join(outputs_path, args.exp_name, "scores")
    json_path = os.path.join(score_path, "result.json")
    if not os.path.exists(json_path):
        print(f"Error: Cannot find json path {json_path}")
        exit(1)
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    final_score, components = compute_final_score(data)
    print_report(final_score, components)