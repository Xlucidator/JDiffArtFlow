baseline


my baseline
```
Steps: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 800/800 [03:56<00:00,  3.40it/s, loss=0.358]Saving LoRA weights to /workspace/outputs/style_ckpt/style_00
Model weights saved in /workspace/outputs/style_ckpt/style_00/pytorch_lora_weights.bin
Steps: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 800/800 [03:56<00:00,  3.38it/s, loss=0.358]
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
