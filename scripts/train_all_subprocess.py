import os
import shutil
import sys
import time
import yaml
import copy
import argparse
from multiprocessing import Process, Queue
import subprocess
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
src_path = project_root / "src"
configs_path = project_root / "configs"
logs_path = project_root / "logs"
outputs_path = project_root / "outputs"

train_logs_path = logs_path / "train"
style_ckpt_path = outputs_path / "style_ckpt"
temp_config_path = configs_path / "temp_configs"
train_script_path = src_path / "run_train.py"

def prepare_dirs():
    dirs_to_reset = [outputs_path, train_logs_path, style_ckpt_path, temp_config_path]
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


def worker_process(gpu_id, task_queue, base_config_dict):
    print(f"[GPU {gpu_id}] Worker Start...")
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    
    log_file = train_logs_path / f"gpu_{gpu_id}.log"
    file_logger = open(log_file, "w", encoding='utf-8')
    console = sys.stdout
    sys.stdout = file_logger
    sys.stderr = file_logger

    while not task_queue.empty():
        try:
            # non blocked get
            style_idx = task_queue.get_nowait()
        except:
            break

        style_str = f"{style_idx:02d}"
        print(f"[GPU {gpu_id}] === Start processing Style {style_str} ===")

        current_config = copy.deepcopy(base_config_dict)
        current_config['data']['instance_data_dir'] = f"./data/train/{style_str}/images"
        current_config['data']['instance_prompt'] = f"style_{style_str}"
        current_config['data']['class_data_dir'] = f"./data/prior/{style_str}"
        current_config['experiment']['output_dir'] = f"./outputs/style_ckpt/style_{style_str}"

        temp_config_file = temp_config_path / f"train_config_{style_str}.yaml"
        with open(temp_config_file, 'w') as f:
            yaml.dump(current_config, f)

        cmd = [
            sys.executable,
            str(train_script_path.resolve()), 
            "--config", f"temp_configs/train_config_{style_str}.yaml"
        ]
        
        try:
            subprocess.run(cmd, check=True, stdout=file_logger, stderr=file_logger)
            print(f"[GPU {gpu_id}] Style {style_str} training completed!")
        except subprocess.CalledProcessError as e:
            print(f"[GPU {gpu_id}] !!! Error in Style {style_str}: {e}")
        finally:
            if temp_config_file.exists():
                os.remove(temp_config_file)

    print(f"[GPU {gpu_id}] All tasks finished. Logs saved to {log_file}", file=console)
    file_logger.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--max_num", type=int, default=15, help="Total number of styles to run (0 to max_num-1)")
    parser.add_argument("-c", "--config", type=str, default="self-tune.yaml")
    # parser.add_argument("-g", "--gpu_count", type=int, default=4, help="Number of GPUs to use")
    args = parser.parse_args()

    # 0. Load base config
    prepare_dirs()
    base_config_path = configs_path / args.config
    print(f"=== Loading base config from {base_config_path} ===")
    base_config_dict = load_yaml(base_config_path)

    # 1. Create task queue
    task_queue = Queue()
    for i in range(args.max_num):
        task_queue.put(i)
    print(f"=== Starting batch training ===\nTotal tasks: {args.max_num}")

    # 2. Start Worker processes
    processes = []
    active_gpu_ids = [1, 2, 3]
    try:
        for gpu_id in active_gpu_ids:
            p = Process(target=worker_process, args=(gpu_id, task_queue, base_config_dict))
            p.start()
            processes.append(p)
            time.sleep(5) # Slightly stagger start times to avoid IO/VRAM spikes & Jittor compile conflicts

        for p in processes:
            p.join()
    except KeyboardInterrupt:
        print("\n=== KeyboardInterrupt detected. Terminating all processes... ===")
        for p in processes:
            if p.is_alive():
                print(f" - Killing Process {p.pid}...")
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

    print("\n=== All training tasks have completed ===")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method('spawn')
    main()