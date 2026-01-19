# 评分脚本目录

使用 `run_enval.py` 包装了官方提供的 `jdiff_checker` 并使输出格式更易于查看；添加了总分计算和报告导出脚本 `eval_score.py`

评分conda环境配置

```bash
conda create -n jdiff_eval python=3.9 -y
conda activate jdiff_eval
pip install -r requirements.txt
```

单独使用：确保 `outputs/` 目录中存在已生成图片目录 `generate` 
```bash
python run_enval.py -n <style_num> # 前 <style_num> 个风格图片评测，生成详细评分于 outputs/scores/ 
python eval_score.py  # 读取分析 outputs/scores/result.json ，给出总分并输出报告于 outputs/score.txt 
```