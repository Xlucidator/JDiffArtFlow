# JDiffArt

风格迁移图像生成模型


## 快速开始

### 环境准备

训练和推理：

使用官方docker镜像，然后创建docker环境
```bash
# project root
./docker_env.sh start
```

评分：

创建conda环境
```bash
conda create -n jdiff_eval python=3.9 -y
conda activate jdiff_eval
pip install -r evaluation/requirements.txt
```

### 训练 & 推理 & 评分

```bash
# project root
bash scripts/auto_pipeline.sh 15
```

然后就可以等待生成