
之前的流程：假定实验名为 `exp_name`
```bash
# 确定跑的数量 要改3个地方 train_all.sh, run_all.py 和 score_api.py

# Docker (jdiffusion), in scripts/ 
cd scripts
MAX_NUM=1 bash train_all.sh   # 获得 style_ckpt
python run_all.py -n 1        # 获得 generate

# Host (jdiff_eval)
python evaluation/run_eval.py -n 1 -e exp_name # 获得 scores
python scripts/eval_score.py -n 1 -e exp_name # 得到总分
```