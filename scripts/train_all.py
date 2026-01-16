import os
import shutil
import sys
import time
import yaml
import copy
import argparse
from multiprocessing import Process, Queue
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
src_path = project_root / "src"
configs_path = project_root / "configs"
logs_path = project_root / "logs"
outputs_path = project_root / "outputs"

BASE_CONFIG_FILE = "self-tune.yaml"
BASE_CONFIG_PATH = configs_path / BASE_CONFIG_FILE

def prepare_dirs():
    dirs_to_reset = [logs_path, outputs_path]
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
    
    log_file = logs_path / f"gpu_{gpu_id}.log"
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

        sys.path.append(str(src_path))
        try:
            from run_train import main as train_task  # type: ignore
        except ImportError:
            sys.path.append(str(src_path))
            from run_train import main as train_task  # type: ignore

        style_str = f"{style_idx:02d}"
        print(f"[GPU {gpu_id}] === Start processing Style {style_str} ===")
        
        try:
            current_config = copy.deepcopy(base_config_dict)

            current_config['data']['instance_data_dir'] = f"./data/train/{style_str}/images"
            current_config['data']['instance_prompt'] = f"style_{style_str}"
            current_config['experiment']['output_dir'] = f"./outputs/style_ckpt/style_{style_str}"

            train_task(yaml_config=current_config)
            print(f"[GPU {gpu_id}] Style {style_str} training completed!")
        except Exception as e:
            print(f"[GPU {gpu_id}] Style {style_str} failed: {e}")
            import traceback
            traceback.print_exc()

    print(f"[GPU {gpu_id}] All tasks finished. Logs saved to {log_file}", file=console)
    file_logger.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--max_num", type=int, default=15, help="Total number of styles to run (0 to max_num-1)")
    # parser.add_argument("-g", "--gpu_count", type=int, default=4, help="Number of GPUs to use")
    args = parser.parse_args()

    # 0. Load base config
    prepare_dirs()
    print(f"=== Loading base config from {BASE_CONFIG_PATH} ===")
    base_config_dict = load_yaml(BASE_CONFIG_PATH)

    # 1. Create task queue
    task_queue = Queue()
    for i in range(args.max_num):
        task_queue.put(i)
    print(f"=== Starting batch training ===\nTotal tasks: {args.max_num}")

    # 2. Start Worker processes
    processes = []
    active_gpu_ids = [2, 3]
    for gpu_id in active_gpu_ids:
        p = Process(target=worker_process, args=(gpu_id, task_queue, base_config_dict))
        p.start()
        processes.append(p)
        time.sleep(2) # Slightly stagger start times to avoid IO/VRAM spikes

    # 3. Wait for all processes to finish
    for p in processes:
        p.join()

    print("\n=== All training tasks have completed ===")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method('spawn')
    main()