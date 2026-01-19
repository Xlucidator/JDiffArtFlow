# 数据集目录

train 目录：训练和推理用数据，A组数据 00 \~ 14 共 15 个风格，每种风格包含 
- `prompt.json` : 25 个推理 prompt
- `images` : 10 张训练用目标风格图

A_gt 目录：评分用目录，但内容其实与 train 一模一样，冗余删去。

prior 目录：开启 Prior Preservation 模式必备，针对每个风格训练和推理 prompt 生成的无风格图片
- 每个风格目录中包含 (10 + 25) * 8 = 280 张 512x512 像素的图片
- 可以使用 `scripts/gen_prior.py` 现场生成，不过尚未 batch 处理生成较慢；也可通过 [云盘](https://cloud.tsinghua.edu.cn/d/6bac317eada140408f9a/) 获取项目训练时用数据，解压并置于此目录中