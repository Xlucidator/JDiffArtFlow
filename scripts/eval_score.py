import json
import argparse
import os
import numpy as np

def compute_final_score(metrics):
    """
    compute_final_score with the formula: (Style Similarity) × (Text Consistency + Image Quality)
    args:
        - metrics (dict): containing 'dino_score', 'lpip_score', 'clip_r_score', 'fid_score', 'clip_iqa_score'
    returns:
        - final_score (float): the final score
        - components  (dict): calculated values of each component (Style, Text, Quality)
    """
    
    ## === 1. Metrics Extract ===
    # high-is-better
    dino = metrics.get('dino_score', 0.0) 
    clip_r = metrics.get('clip_r_score', 0.0)
    clip_iqa = metrics.get('clip_iqa_score', 0.0)
    
    # low-is-better
    lpips = metrics.get('lpip_score', 1.0)      # 0~1 
    fid = metrics.get('fid_score', 100.0)       # >0
    

    ## === 2. Calculate Components ===
    # --- A. Style Similarity ---
    # Determined by LPIPS and DinoV2.
    # - Dino : focuses on "semantic/structural layout", 
    # - LPIPS: focuses on "perceptual/texture details", distance metric -> convert to similarity (1 - LPIPS)
    style_sim_lpip = max(0, 1 - lpips) # Normalize to 0-1
    style_sim_dino = dino
    style_component = 0.5 * style_sim_dino + 0.5 * style_sim_lpip

    # --- B. Text Consistency ---
    # Directly use CLIP-R (Recall)
    text_component = clip_r
    
    # --- C. Images Quality ---
    # Determined by CLIP-IQA and FID.
    # - CLIP-IQA: 0~1 aesthetic quality score
    # - FID: distance metric, lower is better, need normalization
    quality_clip_iaq = clip_iqa
    quality_fid = 1.0 / (fid + 1.0)
    quality_component = 0.8 * quality_clip_iaq + 0.2 * quality_fid
    

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
    parser.add_argument("--result_path", type=str, default="./outputs/baseline/scores")
    args = parser.parse_args()

    json_path = os.path.join(args.result_path, "result.json")
    if not os.path.exists(json_path):
        print(f"Error: Cannot find json path {json_path}")
        exit(1)
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    final_score, components = compute_final_score(data)
    print_report(final_score, components)