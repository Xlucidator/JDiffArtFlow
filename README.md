# JDiffArt

风格迁移图像生成模型


## How to Run

### Train & Generate

准备Docker环境，修正jittor依赖

根据 `scripts` 中说明，添加peft依赖
```bash
pip install accelerate==0.27.2
pip install peft==0.10.0
```

正式训练和生成
```bash
# project root
./docker_env.sh enter # 进入docker环境
cd scripts

bash train_all.sh  # 训练
python run_all.py  # 生成
```

### Evaluation

准备测评环境
```bash
conda create -n jdiff_eval python=3.9 -y
conda activate jdiff_eval
pip install -r requirements.txt
```

正式测评
```bash
conda activate jdiff_eval
cd evaluation
python run_eval.py
```