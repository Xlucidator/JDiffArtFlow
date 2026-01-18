baseline

## 实验 & 优化记录

1. 跑通example中的脚本训练和推理；跑通评分脚本，优化输出格式，计算总分，添加包装脚本
2. 基于train.py，重构出结构清晰的baseline，置于src目录中
3. 设置seed，固定随机性（但没完全固定住）
3. 优化：修改prompt, 动态prompt，"an image of {object} in style_{taskid}"
  - 后面记起来，将train和infer的prompt调成一样的了
4. 优化：提高unet_lora_rank，从4到32，提升显著
5. 优化：为text_encoder旁接lora
  - 使用peft添加，所以在infer时也需要peft添加，然后将te_lora硬注入进去；改了很久，见代码
  - 设置te的学习率为unet的0.1倍
  - 试了te_lora_rank从4到2到0，发现加的越少越好；大概是语义符合性增加比不上风格的缺失；最后设为了0
6. 优化：(参考chyxx项目) 推理时增加训练图的avg底图，自定义新的生成管线 `Img2ImgPipeline` ，用平均底图+prompt一起生成图片；些许显著，把te_lora_rank降至0后，首次总分上0.5
7. 借上一条，重写了run_infer，然后将推理也并行化；之前也将train并行化（scripts中train_all.py+infer_all.py;src中run_train.py和run_infer.py）
8. 优化：训练时实现 with_prior_preservation；生成了280张class图片，和训练用风格图一起，训练时看一张风格图，再看一张class图；生成图片的语义更符合了，质量更高了，不过总分变化不大（感觉也有随机性作梗）
9. 配置了全流程的自动化管线，训练+推理+打分+报告
10. 优化：训练时增加了数据增强：添加了翻转和旋转；同时发现之前的RandomCrop没有起作用（因为1024x1024采样后直接变为512x512，没的RandomeCrop了；变化不大，好像有负优化的趋势，增大max_training_steps看看
11. 再次尝试固定随机性
  - auto_pipeline.bash开始设置
  - dataset 中 iterdir 排序
  - train_all和infer_all改用subprocess发射任务



## Record

```bash
==================================================
METRIC NAME               | SCORE
--------------------------------------------------
dino_score                | 0.164997
hist_score                | 0.829148
fid_score                 | 2.119416
lpip_score                | 0.409496
clip_iqa_score            | 0.758036
clip_r_score              | 0.810667
==================================================

========================================
            Score Report
========================================
1. Style Component  : 0.6009
2. Text Component   : 0.4053
3. Quality Component: 0.4334
----------------------------------------
  Final Score: 0.503971
========================================
```


## Possible Try

### Parameter

1. 增加lora rank : 8, 16, 32。
2. learning rate : rank增加了，可以稍微降低一些lr, 5e-5
3. train step : 500, 700, 800, 900, 1000, 1200
4. resolution : 768

### Framework

1. 训练 Text Encoder (Text Encoder LoRA)
2. 先验保留 (Prior Preservation Loss)
3. 推理技巧：平均色调初始化 (Average Image Init)
4. 推理技巧：Prompt 融合 (Prompt Interpolation)



## Record

style_00

0. change prompt to "style_00":  0.4093
1. use dynamic prompt "an image of {object} in style_00", align run_all prompt:  
2. unet_lora to 32, add text_encoder_lora = 16: 
