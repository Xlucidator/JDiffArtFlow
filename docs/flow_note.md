
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