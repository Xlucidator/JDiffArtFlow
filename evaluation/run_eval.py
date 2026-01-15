import os
import subprocess
import sys
import argparse

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# os.environ["HF_HUB_OFFLINE"] = "1"

current_script_path = os.path.abspath(__file__)
eval_dir = os.path.dirname(current_script_path)
project_root = os.path.dirname(eval_dir)
outputs_dir = os.path.join(project_root, "outputs")

def parse_args():
    parser = argparse.ArgumentParser(description="Run Evaluation Script")
    parser.add_argument("-e", "--exp_name", type=str, default=".")
    parser.add_argument("-n", "--num_styles", type=int, default=15)
    return parser.parse_args()


def run_evaluation(exp_name, num_styles):
    checker_dir = os.path.join(eval_dir, "jdiff_checker")
    script_name = "score_api.py"
    exp_path = os.path.join(outputs_dir, exp_name)
    upload_path = os.path.join(exp_path, "generate")
    scores_path = os.path.join(exp_path, "scores")

    print(f"[Launcher] Begin to run evaluation...")
    print(f"   - working dir: {checker_dir}")
    print(f"   - upload path: {upload_path}")

    if not os.path.exists(upload_path):
        print(f"Error: Upload path not found: {upload_path}")
        return
    
    expected_gt_path = os.path.join(project_root, "data", "A_gt")
    if not os.path.exists(expected_gt_path):
        print(f"Error: Ground truth path not found: {expected_gt_path}")
        print(f'Please ensure that the ground truth data is available at: {expected_gt_path}')
        return
    
    cmd = [
        sys.executable,
        script_name,
        "--upload_path", upload_path,
        "--result_path", scores_path,
        "--ref_style_numbers", str(num_styles)
    ]

    try:
        subprocess.run(cmd, cwd=checker_dir, check=True)
        print(f"[Launcher] Evaluation completed successfully. saved to {scores_path}")
    except subprocess.CalledProcessError as e:
        print(f"[Launcher] Evaluation failed with error: {e}")
    except Exception as e:
        print(f"[Launcher] An unexpected error occurred: {e}")

if __name__ == "__main__":
    args = parse_args()
    run_evaluation(args.exp_name, args.num_styles)