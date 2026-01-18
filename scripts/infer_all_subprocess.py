import os
import shutil
import subprocess
import sys
import yaml
import argparse
import copy
import multiprocessing as mp
import time
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
src_path = project_root / "src"
configs_path = project_root / "configs"
outputs_path = project_root / "outputs"
logs_path = project_root / "logs"

infer_logs_path = logs_path / "infer"
generate_path = outputs_path / "generate"
score_path = outputs_path / "scores"
temp_config_path = configs_path / "temp_configs"
infer_script_path = src_path / "run_infer.py"

def prepare_dirs():
    dirs_to_reset = [infer_logs_path, generate_path, score_path, temp_config_path]
    for dir_path in dirs_to_reset:
        if dir_path.exists():
            print(f"Cleaning: {dir_path}")
            try:
                shutil.rmtree(dir_path)
            except Exception as e:
                print(f"Error cleaning {dir_path}: {e}")

        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"Created:  {dir_path}")


def load_yaml(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)
    
def save_yaml(config_dict, path):
    with open(path, 'w', encoding='utf-8') as file:
        yaml.dump(
            config_dict, 
            file, 
            default_flow_style=False, # block style output, more readable
            sort_keys=False,          # keep order
            allow_unicode=True        # allow unicode characters
        )


def worker_process(gpu_id, task_queue, base_config_dict):
    print(f"[GPU {gpu_id}] Worker Start...")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

    log_file = infer_logs_path / f"gpu_{gpu_id}.log"
    file_logger = open(log_file, "w", encoding='utf-8')
    console = sys.stdout
    sys.stdout = file_logger
    sys.stderr = file_logger
    
    while not task_queue.empty():
        try:
            # non blocked get
            style_idx = task_queue.get_nowait()
        except Exception:
            break

        # sys.path.append(str(src_path))
        # try:
        #     from run_infer import run_single_task as infer_task  # type: ignore
        # except ImportError as e:
        #     print(f"[GPU {gpu_id}] Error importing run_infer module: {e}")
        #     exit(1)

        style_str = f"{style_idx:02d}"
        print(f"[GPU {gpu_id}] === Start processing Style {style_str} ===")
        
        current_config = copy.deepcopy(base_config_dict)
        # TODO: (option) modify config for each style

        temp_config_file = temp_config_path / f"infer_config_{style_str}.yaml"
        with open(temp_config_file, 'w') as f:
            yaml.dump(current_config, f)

        cmd = [
            sys.executable,
            str(infer_script_path.resolve()),
            "--taskid", style_str,
            "--config", f"temp_configs/infer_config_{style_str}.yaml"
        ]

        try:
            subprocess.run(cmd, check=True, stdout=file_logger, stderr=file_logger)
            print(f"[GPU {gpu_id}] Style {style_str} inference completed!")
        except subprocess.CalledProcessError as e:
            print(f"[GPU {gpu_id}] !!! Error in Style {style_str}: {e}")
        finally:
            if temp_config_file.exists():
                os.remove(temp_config_file)
    
    print(f"[GPU {gpu_id}] All tasks finished. Logs saved to {log_file}", file=console)
    file_logger.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num_styles", type=int, default=15, help="Total number of styles (0 to n-1)")
    parser.add_argument("-e", "--exp_name", type=str, default=".")  # useless now
    parser.add_argument("-c", "--config", type=str, default="self-tune.yaml")
    # parser.add_argument("--gpus", type=str, default="0,1,2,3", help="Comma separated list of GPU ids to use")
    args = parser.parse_args()

    # 0. Load base config
    prepare_dirs()
    base_config_path = configs_path / args.config
    print(f"=== Loading base config from {base_config_path} ===")
    base_config_dict = load_yaml(base_config_path)

    # 1. Create task queue
    task_queue = mp.Queue()
    for i in range(args.num_styles):
        task_queue.put(i)
    print(f"=== Starting Multi-GPU Inference ===")
    print(f"Total Tasks: {args.num_styles}")

    # 2. Start Worker processes
    active_gpu_ids = [1, 2, 3]
    print(f"Use GPUs: {active_gpu_ids}")
    processes = []

    try:
        for gpu_id in active_gpu_ids:
            p = mp.Process(target=worker_process, args=(gpu_id, task_queue, base_config_dict))
            p.start()
            processes.append(p)
            time.sleep(5)  # Slightly stagger start times to avoid IO/VRAM spikes

        for p in processes:
            p.join()
    except KeyboardInterrupt:
        print("\n=== KeyboardInterrupt detected. Terminating all processes... ===")
        for p in processes:
            if p.is_alive():
                p.terminate()
                p.join()
        print("All processes terminated.")
        exit(1)
    except Exception as e:
        print(f"[Manager] x Unexpected Error: {e}")
        for p in processes:
            if p.is_alive():
                p.terminate()
        exit(1)
    
    
    print("=== All inference tasks have completed ===")

if __name__ == "__main__":
    mp.set_start_method('spawn')
    main()