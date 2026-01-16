修改权限
```bash
sudo chown -R $(id -u):$(id -g) ./outputs
```


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


原本的train.py
```python
def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str)

def parse_args(input_args=None)

class DreamBoothDataset(Dataset):
    def __init__()
    def __len__(self)
    def __getitem__(self, index)

def collate_fn(examples, with_prior_preservation=False)

# Not used
# class PromptDataset(Dataset):
#     def __init__(self, prompt, num_samples)
#     def __len__(self)
#     def __getitem__(self, index)

def tokenize_prompt(tokenizer, prompt, tokenizer_max_length=None)

def encode_prompt(text_encoder, input_ids, attention_mask, text_encoder_use_attention_mask=None)
```


原始开头输出
```
{'timestep_spacing', 'clip_sample_range', 'thresholding', 'dynamic_thresholding_ratio', 'variance_type', 'rescale_betas_zero_snr', 'sample_max_value'} was not found in config. Values will be initialized to default values.
{'scaling_factor', 'force_upcast'} was not found in config. Values will be initialized to default values.
All model checkpoint weights were used when initializing AutoencoderKL.

All the weights of AutoencoderKL were initialized from the model checkpoint at Charles-Elena/stable-diffusion-2-1.
If your task is similar to the task the model of the checkpoint was trained on, you can already use AutoencoderKL for predictions without further training.
{'projection_class_embeddings_input_dim', 'time_embedding_type', 'time_embedding_act_fn', 'mid_block_only_cross_attention', 'addition_embed_type', 'time_embedding_dim', 'resnet_out_scale_factor', 'cross_attention_norm', 'attention_type', 'addition_time_embed_dim', 'timestep_post_act', 'num_attention_heads', 'resnet_skip_time_act', 'class_embeddings_concat', 'resnet_time_scale_shift', 'conv_out_kernel', 'encoder_hid_dim_type', 'mid_block_type', 'conv_in_kernel', 'dropout', 'transformer_layers_per_block', 'reverse_transformer_layers_per_block', 'class_embed_type', 'time_cond_proj_dim', 'addition_embed_type_num_heads', 'encoder_hid_dim'} was not found in config. Values will be initialized to default values.
All model checkpoint weights were used when initializing UNet2DConditionModel.

All the weights of UNet2DConditionModel were initialized from the model checkpoint at Charles-Elena/stable-diffusion-2-1.
If your task is similar to the task the model of the checkpoint was trained on, you can already use UNet2DConditionModel for predictions without further training.
lr: 0.0001
```

一般loss
```bash
Steps: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 800/800 [03:55<00:00,  3.43it/s, loss=0.552]Saving LoRA weights to /workspace/outputs/style_ckpt/style_00
Model weights saved in /workspace/outputs/style_ckpt/style_00/pytorch_lora_weights.bin
Steps: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 800/800 [03:55<00:00,  3.39it/s, loss=0.552
```