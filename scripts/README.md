# 批处理脚本目录

自动化流水线脚本 `auto_pipeline.sh` 调用链

```bash
[host] `bash auto_pipeline -n <style_num>`
  ├──> [docker] `python scripts/train_all_subprocess.py -n <style_num>` (or train_all.py)
  ├──> [docker] `python scripts/infer_all_subprocess.py -n <style_num>` (or infer_all.py)
  ├──> [docker] `chown ...` change permisson of generated files in outputs 
  ├──> [host]   `python evaluation/run_eval.py -n <style_num>`
  └──> [host]   `python evaluation/eval_score.py`
```

---

`[train/infer]_all_subprocess.py` 是 `[train/infer]_all.py` 的 subprocess 隔离版，通过进程隔离机制，强制为每个风格任务重置内存与 CUDA 上下文，旨在杜绝任务间的状态污染与条件依赖。然而实验表明，即便实现了环境隔离，仍无法彻底消除结果的随机性。

由于需要反复读写超参数 `yaml` 配置，subprocess 隔离版的运行速度慢于原版，鉴于实验尚无法根除计算结果的随机性，也可替换会原版脚本以追求更高效的训练推理流程。

---

`gen_prior.py` 用于生成 Prior Preservation 训练功能所需使用的无风格图片。 `check_keys.py` 可用于检查模型参数key名称。


----

`deprecated` 目录中为原始 Baseline 训练和推理脚本；推理用 `run_all.py` 在此后训练优化过程仍被继续使用了很长一段时间，因而存在大幅度的调整，但依旧兼容旧版 Baseline 功能。