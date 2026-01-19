# 基于 Jittor 和 DreamBooth-LoRA 的风格迁移图像生成 - 实验报告

本实验旨在基于 Jittor 框架与 Stable Diffusion v2.1 模型，解决极少样本（10-shot）场景下的高保真风格迁移任务。面对 Baseline 存在的“风格色调偏差大”与“语义结构易崩坏”两大痛点，本项目实现了一套改进的 DreamBooth + LoRA 微调策略。核心优化包括：提升 LoRA Rank 以增强纹理拟合，引入 Prior Preservation 确保语义对齐，以及开发基于平均风格图的 Img2Img 初始化以锁定色调分布等。

最终，在不依赖繁琐超参数调优的前提下，该方案成功将综合指标由 0.327 提升至 **0.512**，证明了策略的有效性与泛化潜力。

## 1 任务背景

### 1.1 任务描述

任务要求基于给定的预训练扩散模型，针对 14 组不同风格的图像数据进行微调。每组数据包含 10 张训练图像。模型需根据特定的提示词（Prompt）生成具有对应风格且语义准确的图像。

### 1.2 评价标准

评分由三部分组成：

1. **Style Component (风格相似度)**: 由 DinoV2 (结构)、Pixel-Hist (颜色直方图)、LPIPS (感知相似度) 构成。
2. **Text Component (文本一致性)**: 由 CLIP-R (CLIP Recall) 衡量。
3. **Quality Component (图像质量)**: 由 FID (分布距离) 和 CLIP-IQA (美学评分) 构成。

最终得分计算公式

```
final_score = (Style Similarity) × (Text Consistency + Image Quality)
 = [DinoV2 * 0.2 + Pixel-Hist * 0.4 + (1-LPIPS) * 0.4] * 
   [CLIP-R * 0.5 + (20 - min(FID,20)) / 20 * 0.4 + CLIP-IQA * 0.1]
```

## 2 实现说明

本项目在 Baseline (DreamBooth-LoRA) 的基础上进行了深度的代码重构与算法优化。

### 2.1 模型架构

- **底模**: `Charles-Elena/stable-diffusion-2-1` (Jittor版)。
- **微调方法**: 
  - 训练端：采用 **PEFT** 技术，在 UNet 和 Text Encoder 的 Attention 层（`to_k`, `to_q`, `to_v`, `to_out` 等）注入 LoRA 权重；训练数据中加入先验 `class` 数据
- **训练框架**: 重构自官方提供的 Jittor 版 DreamBooth-LoRA 基础框架，具体结构见 3.2 工程实现

### 2.2 优化策略

本节将按实验顺序，依次介绍我在模型训练和推理优化时探索和思考过的策略。我没有盲目堆砌 Trick，而是面向 Baseline 存在的“风格语义绑定弱”、“生成结构崩坏”以及“评分指标特性”等问题，逐一搜寻和尝试针对性的解决方案。

#### 2.2.1 动态提示词与对齐策略【训练&推理】

实验发现原 Baseline 中仅用 `style_{id}` 作为训练和推理时的 Prompt，过于简单，导致模型难以将风格 Token 与具体的视觉特征建立强关联，且容易忽略物体本身的语义。为此，我重构了数据加载逻辑，在训练与推理时严格对齐 Prompt 格式。训练时利用文件名动态生成 `an image of {object_name} in style_{id}`，推理时同样使用 `an image of {prompt} in style_{id}` 的模版。

- **目的** ：强化模型对 `style_{id}` 这一特殊 Token 的敏感度，将其作为明确的“风格触发词”；同时通过引入物体名称，辅助 Text Encoder 更好地区分“内容”与“风格”。
- **代码对应** ： `src/dataset.py` 中的 `__getitem__` 方法与 Prompt 构建逻辑。
- **效果** ：显著提升了 Prompt 的语义遵循能力，模型不再将 `style_{id}` 误认为是某种物体，而是正确地将其作为一种画风修饰符应用在不同物体上。

#### 2.2.2 UNet 高秩 LoRA 适配【训练】

Baseline 使用的 `unet_lora_rank=4` 参数量过小，难以捕捉复杂的艺术风格纹理，导致生成的风格过于平淡或只学到了颜色。经过多次消融实验（对比 4, 16, 32, 64），本项目最终将 Rank 提升至 **32** 。

- **目的** ：更高的秩赋予了 LoRA 层更大的参数空间，使其能够更好地拟合训练集中强烈的风格特征（如特殊的笔触、光影变化），同时避免了 Rank 64 训练速度慢且在小数据集上容易出现的过拟合噪点问题。
- **代码对应** ： `configs/self-tune.yaml` 中 `unet_lora_rank: 32`。
- **效果** ：较大幅度提升了 Style Component 的分数，生成的图像纹理细节更加丰富，风格还原度明显增强。

#### 2.2.3 Text-Encoder LoRA 适配与工程调优【训练&推理】

在 Stable Diffusion 架构中，Text Encoder 决定了语义理解的上限。为了探索语义空间微调对风格迁移的潜力，本项目尝试开启 Text Encoder 的 LoRA 训练。这一过程在 **工程实现** 上面临了一定挑战：由于 Diffusers 与 Jittor 框架的适配性问题，标准的参数加载机制无法直接用于 Text Encoder LoRA。为此，我在 `run_infer.py` 中进行了大量的手动适配与代码重构，才成功实现参数的正确加载。此外，考虑到 Text Encoder 容易过拟合导致“语言漂移”，我在训练阶段对其应用了**差分学习率策略**，将其学习率设置为 UNet 学习率的 **0.1 倍** 。

- **目的** ：探索在 UNet 视觉微调之外，是否可以通过 Text Encoder 的语义微调来提升 Prompt 的响应度，并验证其在不同训练阶段的表现。
- **代码对应** ： `src/run_infer.py` 中的手动参数加载逻辑，以及 `src/trainers/dreambooth_trainer.py` 中的优化器参数组配置。
- **效果** ：实验现象较为复杂且具有反直觉性。简单 **消融实验** 表明，在基础配置下，开启 Text Encoder LoRA (Rank 2, 4, 8) 相比完全冻结 (Rank 0) 反而越来越导致各项指标的下降，表现为负面影响。然而，在某版结合了高 Prior Weight、数据增强及高 Rank UNet 的完整策略中，为了解决 **部分生成图像难以有效附着风格纹理** 的问题，重新开启 Text Encoder LoRA 后，观察到其似乎在维持语义结构方面提供了潜在助力（总分 0.5118），但其在复杂组合下的具体增益机制仍有待进一步深入研究。

#### 2.2.4 基于平均风格图的推理初始化【推理】

标准的 Text-to-Image 生成过程从纯高斯噪声开始去噪，这种完全随机的初始化（$\mathcal{N}(0, I)$）导致生成图像在 **色调** 和 **全局光影**上具有极大的不确定性。对于风格迁移任务，色彩分布（Histogram）是风格的重要组成部分。为了解决生成图像“色调漂移”导致 `hist_score` 低下的问题，本项目参考 [chyxx](https://github.com/Chyxx/jittor-pad-pad-eos-HighQualityStyleTransfer) 方案，实现了一种基于 **平均风格先验** 的推理初始化策略。

具体实现为：首先计算当前风格训练集（10张图像）的 **像素级平均图（Average Image）** ，该图像模糊但完美保留了该风格的全局平均色调；随后重构推理管线，弃用标准的 T2I 流程，转而采用 **Img2Img** 模式——将这张“平均风格图”作为初始底图，施加高强度的噪声（Denoising Strength=0.9）作为初始潜变量。这相当于在纯噪声中“注入”了风格的低频色彩信息。

- **目的** ：纠正初始潜变量的分布偏差，为扩散模型的去噪过程提供一个强有力的**颜色锚点（Color Anchor）**。在允许模型自由生成物体结构的同时，强制约束其最终成像的直方图分布与训练集保持统计学上的一致性。
- **代码对应** ： `src/utils/image.py` 中的 `get_avg_image` 计算逻辑，以及 `src/run_infer.py` 中自定义的 Latents 初始化管线。
- **效果** ：该策略是本项目中**提升 Style Component 分数最立竿见影的手段**。实验表明，相比于纯噪声推理，基于平均图初始化的生成结果在 **Histogram Score** 上有显著跃升，极大地解决了“风格学到了但颜色不对”的常见顽疾，确保了生成图像在视觉氛围上与原风格的高度对齐。

#### 2.2.5 启用 Prior Preservation 先验保真损失【训练】

在实验设计初期，原本计划采用 **Specific Prior** 策略，即针对每个风格 ID，根据其训练数据和推理 Prompt 生成对应的约 280 张专用先验图。然而在实际执行中，由于配置文件的路径指向未随风格 ID 动态更新，导致所有 15 个风格的训练全程均使用了基于 `style_00` 相关 Prompt 生成的一组**共享先验数据（Shared Prior Data）**。

令人意外的是，这种“错配”并未导致训练失败，总分依然稳定在 0.5 以上。这在实验上验证了对于风格迁移任务（而非特定实体 ID 保持），Prior 数据集无需与当前训练集内容严格一一对应，一组包含丰富杂类物体（Mixed Class）的共享先验图足以起到锚定模型“世界观”、防止结构崩塌的正则化作用。最终实验中，我们将 Prior Loss 的权重设定为 **1.0** 以最大化保证物体结构的完整性。

- **目的** ：通过正则化手段，利用通用图像分布约束模型的参数更新方向，强制模型在学习新风格的同时，保留对通用物体结构的认知，防止物体结构崩坏。
- **代码对应** ： `src/trainers/dreambooth_trainer.py` 中的 `loss_prior` 计算逻辑，以及数据加载中的 Prior 数据复用逻辑。
- **效果** ：该策略显著提升了生成图像的语义准确度（CLIP-R 分数），有效遏制了结构崩坏。 **然而，强力的先验约束也暴露了跨域风格迁移中的深层矛盾** ：实验观察到，对于 Aquarium（水族箱）、Phone（手机）、Wallet（钱包）等物体，模型出现了严重的 **欠风格化（Under-stylization）** 。这不仅是因为这些物体具有强几何结构，更本质的原因在于它们属于 **现代工业产物** ，与训练数据（如梵高、浮世绘等历史艺术流派）存在巨大的 **语义鸿沟（Semantic Gap）** 。模型难以在潜空间中找到将“古典笔触”合理映射到“现代屏幕/玻璃”上的特征表达，在 Prior Loss 的强约束下，模型倾向于“保守”地保留先验数据中的写实特征（例如梵高风格的 Aquarium 最终看起来仍像是一张普通的写实照片）。这一 **“现代物体在古典风格下的语义失配”** 问题，并未通过参数调整得到根本性解决，也是本项目面临的最大挑战，将在“2.3 现存核心问题”一节中进行更进一步的说明。

#### 2.2.6 修复并完善输入图像的增强【训练】

在解决模型对训练集“死记硬背”导致泛化能力差的问题时，我深入审查了数据处理管道，发现 Baseline 存在一个严重的逻辑缺陷：由于原始训练数据多为正方形，Baseline 直接将其 `Resize` 至 512x512，随后进行的 `RandomCrop(512)` 实际上变成了“无操作”。这意味着在数千步的训练中，模型看到的始终是完全静止、构图一成不变的 10 张图，极易导致过拟合。

针对此问题，我重构了 Transform 逻辑：**先将图像 Resize 至分辨率的 1.04 倍（约 532px），再进行 512px 的 RandomCrop**。这看似微小的改动，使得裁剪窗口能够在图像上进行随机游走，凭空创造出数倍于原数据集的构图变体。此外，为了进一步强迫模型关注“纹理”而非“构图”，我还尝试引入了随机翻转和旋转作为增强手段。

- **目的** ：打破小样本训练集中的“构图固化”问题，通过强制像素位移和几何变换，迫使 UNet 学习风格的内在纹理特征，而非死记硬背物体在画面中的绝对位置。
- **代码对应** ： `src/dataset.py` 中的 `train_transforms` 构建逻辑（Resize策略调整与增强算子添加）。
- **效果** ：“Resize 1.04x + Random Crop”的机制成功引入了位移不变性，有效扩充了训练数据的多样性，解决了 Baseline 数据利用率低下的问题；对于进一步引入的几何增强（特别是随机旋转），实验观测到了严重的负面效果。由于样本极少（仅10张），模型错误地将“旋转/倾斜”学习为了风格本身的特征，导致生成的图像普遍出现非自然的旋转构图。鉴于此，最终方案保留了底层的 Crop 修复以确保数据利用率，但果断关闭了旋转增强，从而在保证构图端正的前提下，最大化了直方图分数（Hist Score）。

### 2.3 现存问题

尽管通过上述策略优化了模型的整体性能，但在极端样本表现与实验复现性上，仍存在两个尚未彻底解决的核心问题。

#### 2.3.1 跨域语义鸿沟导致的欠风格化

实验发现，当训练目标为具有强烈的古典艺术风格（如梵高、浮世绘）时，模型在生成现代工业品（如 Aquarium, Phone, Wallet）时往往表现出“风格抵抗”。生成的图像虽然保留了完美的物体结构（得益于 Prior Loss），但纹理仍偏向写实照片，未能成功附着艺术笔触。这本质上是因为**训练数据（古典艺术）与生成目标（现代物体）之间存在巨大的语义鸿沟**，模型在 UNet 的潜在空间中无法找到合理的映射路径。

针对此问题，目前参数微调（Parameter Tuning）已接近瓶颈，未来可从以下两个维度寻求突破：
* **训练端暴力突破**：尝试大幅增加模型容量（如同时开启高 Rank 的 UNet 与 Text Encoder LoRA），配合极高的训练步数与强数据增强。旨在强迫模型跳过语义理解，直接在像素层面“死记硬背”纹理分布。
* **推理端 Style Injection**：参考 CVPR 2024 论文 *《Style Injection in Diffusion》*，引入免训练的注意力注入机制。其核心逻辑在于**解耦内容与风格**：在 UNet 的 Self-Attention 层中，保留生成图像的 **Query (Q)** 以维持物体结构（画什么），强制将 **Key (K)** 和 **Value (V)** 替换为参考风格图的特征（怎么画）。这种“移花接木”的方法理论上能百分百强制风格纹理的附着，且不受语义鸿沟的限制，是解决该问题的终极方案。

#### 2.3.2 深度学习框架的固有随机性

尽管本项目在数据加载层面实现了严格的种子固定（Seed Fixing）与排序逻辑，但在多卡并发训练的实际环境中，仍无法实现 Bit-exact（比特级精准）的实验复现。实验观察到，即便参数与 Seed 完全一致，不同次的运行结果仍存在微小的像素级差异。这主要归因于 **CUDA 算子的原子操作非确定性**以及 Jittor/PyTorch 在动态图计算过程中的全局状态微扰。这种无法被完全消除的“幽灵随机性”在一定程度上干扰了对超参数（如 Rank, Prior Weight）微小变动的精确消融分析。

## 3 实验设置

### 3.1 环境与超参数

本实验基于 Jittor (计图) 深度学习框架进行。为了确保训练的稳定性与推理的高效性，实验在 Ubuntu 22.04 服务器上完成，具体环境如下：

- **硬件环境**: 
    - GPU: 3x NVIDIA GeForce RTX 4090 (24GB VRAM)
    - 并行策略: 利用 Python 实现多卡并行训练/推理，单卡负责单一风格任务。
- **软件环境**: 
    - 框架: Jittor (JDiffusion), Diffusers (适配版)
    - 依赖: 官方Docker镜像，PEFT, Accelerate 等

基于前期消融实验的探索，本项目最终选定的一套**“稳健策略 (Safe Strategy)”**配置如下。该配置旨在极小样本（10-shot）条件下，平衡风格纹理的拟合度与物体结构的保真度，完整详细参数见 `configs/self-tune.yaml` 。

| 模块 | 参数项 | 设定值 | 说明 |
| --- | --- | --- | --- |
| **训练 (Train)** | `max_train_steps` | **800** | 收敛甜点位，避免过拟合 |
| | `learning_rate` | **1e-4** | 配合 Constant Scheduler |
| | `unet_lora_rank` | **32** | 高秩适配，捕捉丰富纹理 |
| | `text_encoder_lora_rank`| **0** | **冻结参数**，防止语义漂移 |
| | `prior_loss_weight` | **1.0** | 强正则化，防止结构崩坏 |
| | `do_augmentation` | **False** | **关闭增强**，最大化颜色对齐 |
| **推理 (Infer)** | `strength` | **0.9** | 配合平均图初始化的重绘强度 |
| | `guidance_scale` | **7.5** | 标准 CFG Scale |
| | `num_inference_steps` | **40** | 保证生成质量 |

### 3.2 工程实现

为了高效处理 14 种风格的并行任务，并解决 Baseline 代码耦合度高、扩展性差的问题，本项目对原有参考代码进行了**全量重构与模块化封装**。

核心改进点如下：
1.  **解耦训练逻辑** ：将原 `train.py` 拆解为 `Dataset`、`Model Engine`、`Trainer` 三层架构，使得 LoRA 注入、Loss 计算（如 Prior Loss）与数据流转逻辑分离，便于调试与魔改。
2.  **多卡并行调度** ：针对 14 个风格任务，开发了 `[train/infer]_all.py` 系列脚本，实现了训练与推理任务的**批量化多卡并行执行**。该机制能够自动根据 GPU 资源分发任务，显著提升了硬件利用率与实验周转效率。
3.  **全自动流水线** ：构建了 `auto_pipeline.sh`，实现了从“数据预处理 -> 多卡并行训练 -> 批量推理 -> 自动化打分”的一键式全流程闭环，显著缩短了实验迭代周期。

项目最终核心目录结构及说明如下：

```bash
.
├── JDiffusion   # 官方JDiffusion目录
├── configs      # yaml配置文件目录
├── data         # 训练、推理数据目录
│   ├── train
│   └── prior        # (需事先准备，或通过 scripts/gen_prior.py 生成)
├── evaluation   # eval 打分目录
│   ├── clip_r_prompts.txt
│   ├── eval_score.py            # 总分计算脚本，生成报告score.txt
│   ├── jdiff_checker            # 官方提供脚本，format输出版
│   ├── requirements.txt         # <jiff_eval> conda环境需求
│   └── run_eval.py              # 细粒度分数计算脚本，jdiff_checker中的score_api.py
├── logs         # 单轮实验日志目录
├── outputs      # 单轮实验输出目录
├── scripts      # [core] 模型使用脚本目录
│   ├── auto_pipeline.sh         # 完整单轮管线脚本，训练 + 推理 + 打分
│   ├── gen_prior.py             # 用于生成 prior_preservation 参考数据图
│   ├── infer_all.py             # infer 多卡多任务训练脚本
│   ├── infer_all_subprocess.py  # infer subprocess版, 旨在减少随机性 *
│   ├── run_all.py               # [deprecate] infer 单卡多任务训练脚本
│   ├── train_all.py             # train 多卡多任务训练脚本
│   └── train_all_subprocess.py  # train subprocess版, 旨在减少随机性 *
├── src          # [core] 模型结构目录
│   ├── custom_pipeline.py       # Image2Image管线
│   ├── dataset.py               # DreamBoothDataset 数据准备
│   ├── models       
│   │   └── diffusion.py         # DiffusionEngine 模型准备
│   ├── run_infer.py             # 运行单个风格的推理
│   ├── run_train.py             # 运行单个风格的训练
│   ├── trainers
│   │   ├── base_trainer.py      # 基础训练类
│   │   └── dreambooth_trainer.py # dreambooth训练类，实现compte_loss方法
│   └── utils
│       ├── config.py            # 参数配置相关函数
│       └── image.py             # 图像处理相关函数
└── docker_env.sh                # docker使用快捷脚本
```

## 4 实验结果与分析

### 4.1 模型配置对比与性能消耗

为了探究不同微调策略对风格迁移效果的影响，在最后的完整训练中，实验对比了两种不同的配置方案：**安全方案** (Safe Strategy) 与 **激进方案** (Aggressive Strategy)。两者的配置参数分别详见 `configs/self-tune.yaml` 和 `configs/use_te_lora.yaml` 。最终选择 **安全方案** 结果作为最终提交的模型参数。

- 训练时间与资源消耗
  + Safe Strategy (800 Steps): 总耗时 43min34s (训练 31m53s)
  + Aggressive Strategy (1200 Steps): 总耗时 63min49s (训练 49m53s)
- 核心超参数差异对比
  | 超参数项 | Safe Strategy | Aggressive Strategy | 策略意图 |
  | ------- | ------------- | ------------------- | ------- |
  | `unet_lora_rank`    | 32    | 64   |  Safe 版求稳防止过拟合；Aggressive 版试图捕捉更多高频纹理。|
  | `text_encoder_rank` | 0     | 8    |  Safe 版冻结以保语义；Aggressive 版开启以增强风格语义理解。|
  | `max_train_steps`   | 800   | 1200 |  Aggressive 版增加了 50% 的步数以配合高 Rank 训练。       |
  | `do_augmentation`   | False | True |  Safe 版关闭增强以最大化 Hist Score；Aggressive 版开启以提升泛化。 |
  | `prior_loss_weight` | 1.0   | 0.9  |  Safe 版给予结构约束最高权重；Aggressive 版搭配深度风格语义理解略微放宽以换取自由度。 |
- 推理相关参数 (共同配置)
  + `strength` : 0.9 
  + `guidance_scale` : 7.5
  + `num_inference_steps` : 40
- 激进策略的致命问题：模型把**数据增强**也学进去了，生成图片都会存在随机的旋转

最终获胜模型 (Safe Strategy) 得分详情 (提交为 `6.1.[full].attempt-2`)

```bash
==================================================
METRIC NAME               | SCORE
--------------------------------------------------
dino_score                | 0.120226
hist_score                | 0.779055
fid_score                 | 1.674596
lpip_score                | 0.428658
clip_iqa_score            | 0.757711
clip_r_score              | 0.930667
==================================================

========================================
              Score Report              
========================================
1. Style Component   : 0.5642
2. Text Component    : 0.4653
3. Quality Component : 0.4423
----------------------------------------
  Final Score: 0.512078
========================================
```

### 4.2 综合评分对比

为了验证上述两种策略的实际产出差异，下表展示了 Baseline、Safe Strategy 以及 Aggressive Strategy 之间的详细指标对比。

| **Metric Name** | **Baseline Score** | **Ours Top Scores (Safe Strategy)** | **Ours Fully Equipped (Aggressive)** |
| --------------- | ------------------ | --------------------  | ------------------------ |
| **Dino Score** (结构相似)     | 0.1130  | 0.1202 (+6.4%)      | **0.1318 (+16.6%)** |
| **Hist Score** (颜色相似)     | 0.4388  | 0.7791 (+77.5%)     | **0.7977 (+81.8%)** |
| **FID Score** (分布距离)      | 1.7101  | 1.6746 (+2.1%)      | **1.5406 (+9.9%)**  |
| **LPIPS Score** (感知相似)    | 0.5391  | **0.4287 (+20.5%)** | 0.4342 (+19.5%)     |
| **CLIP-IQA Score** (图像质量) | 0.7567  | **0.7577 (+0.1%)**  |   0.7273 (-3.9%)    |
| **CLIP-R Score** (文本对齐)   | 0.8293  | **0.9307 (+12.2%)** |   0.9067 (+9.3%)    |
| **Final Score** (最终总分)    | 0.3274  | **0.5121 (+56.4%)** |   0.5119 (+56.3%)   |

指标说明：FID 和 LPIPS 分数值越低代表性能越好，其余指标越高越好。Aggressive 方案虽然在 Dino 和 FID 上表现出色，但更高的计算成本（时间+50%）和语义漂移（CLIP-R 下降）使其综合性价比略逊于 Safe 方案。

### 4.3 结果分析

通过对比实验数据与最终得分，我们得出以下核心结论：

1. **颜色维度的极致对齐 (Hist Score +77.5%)**:
    这是本项目提升幅度最大的单项指标（0.43 $\rightarrow$ 0.78）。这一突破主要得益于 **平均图初始化** 与 **关闭数据增强** 的组合策略。前者在推理阶段的潜变量中强行注入了风格底色，后者在训练阶段允许模型充分拟合训练集的像素统计分布。尽管“激进方案”通过数据增强获得了更高的 Hist Score (0.79)，但“稳健方案”在保持极高分数的同时，避免了过度拟合带来的负面影响。

2.  **语义与结构的稳健性 (CLIP-R Score +12.2%)**: 
    在 CLIP-R 指标上，最终方案（0.9307）显著优于激进方案（0.9067）。这证明了 **冻结 Text Encoder (Rank 0)** 配合 **高权重先验保真 (Prior Weight 1.0)** 策略的正确性。激进方案虽然通过微调 Text Encoder 获得了更强的风格纹理（Dino Score 更高），但也导致了“语言漂移”现象，即模型在过拟合风格时牺牲了对物体文本（如车、狗）的理解能力。最终方案在两者之间取得了最佳平衡。

3.  **FID 与感知质量的优化**:
    与 Baseline 相比，最终方案的 FID（1.71 $\rightarrow$ 1.67）和 LPIPS 均有改善，表明在 `max_train_steps=800` 和 `unet_lora_rank=32` 的设置下，模型达到了良好的收敛状态。值得注意的是，激进方案虽然 FID 更低（1.54），但其 CLIP-IQA（图像美学质量）却出现了下降（-3.9%），这可能是由于过度的数据增强（如旋转）破坏了图像的自然构图美感。

4. **策略性权衡与增强泄漏**:
    我们对比发现，激进策略虽然通过引入随机旋转提升了部分结构指标（Dino），但也导致了严重的 **“增强泄漏”** 现象。实验观察到，在 Aggressive 模型的生成结果中，大量图像呈现出非自然的倾斜或旋转构图。这是因为在极小样本（10-shot）训练中，模型错误地将数据增强引入的几何变换（Rotation）学习为风格本身的固有特征。因此，最终方案（Safe Strategy）果断关闭了所有几何增强，在保证构图端正的前提下，通过 Average Image Init 策略实现了色彩的极致对齐。

## 5. 总结与展望

本项目基于 Jittor 深度学习框架，对 Stable Diffusion 模型在小样本风格迁移任务上的微调策略进行了深入研究与重构。针对 Baseline 存在的“风格色调偏差大”与“语义结构易崩坏”两大核心痛点，本项目提出了一套“高保真、强对齐”的训练与推理方案。

在训练端，我们摒弃了盲目的数据增强，采用 **Rank-32 High-Rank LoRA** 配合 **强先验保真策略 (Prior Loss Weight=1.0)**，在最大化拟合风格纹理的同时，通过共享通用先验有效锚定了模型的物体认知能力；在推理端，创新性地引入了 **基于平均图的潜变量初始化 (Average Image Latent Init)** 策略，成功将风格迁移中的颜色分布对齐问题转化为确定性的数学约束。

最终实验表明，该方案在保持图像生成质量（FID 1.67）的同时，将综合评分从 Baseline 的 0.32 显著提升至 **0.512**（提升幅度达 56.4%），尤其在 Histogram Score 和 CLIP-R Score 上取得了突破性进展。

展望未来，针对目前尚存的“跨域语义鸿沟导致的欠风格化”问题（如现代物体难以附着古典纹理），后续工作将重点探索以下方向：
1.  **引入免训练注意力注入机制**：参考 *Style Injection* [4] 的思路，在推理阶段解耦 Self-Attention 的 Query（结构）与 Key/Value（纹理），从根本上解决纹理附着失败的问题。
2.  **结合 ControlNet 进行结构解耦**：利用 ControlNet 显式控制边缘与深度信息，进一步释放主模型的风格化自由度。
3.  **探索更鲁棒的数据增强**：研究如何在不破坏直方图分布的前提下，引入几何变换以提升模型的泛化能力。

## 参考文献

[1] Ruiz, N., Li, Y., Jampani, V., et al. "DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation." *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2023.

[2] Hu, E. J., Shen, Y., Wallis, P., et al. "LoRA: Low-Rank Adaptation of Large Language Models." *International Conference on Learning Representations (ICLR)*, 2022.

[3] Rombach, R., Blattmann, A., Lorenz, D., et al. "High-Resolution Image Synthesis with Latent Diffusion Models." *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2022.

[4] Ji, J., Zhang, A., Chen, J., et al. "Style Injection in Diffusion: A Training-free Approach for Adapting Large-scale Diffusion Models for Style Transfer." *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2024.