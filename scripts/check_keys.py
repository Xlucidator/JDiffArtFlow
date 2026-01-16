import jittor as jt

weight_file = "../outputs/style_ckpt/style_00/pytorch_lora_weights.bin"
state_dict = jt.load(weight_file)

all_keys = list(state_dict.keys())

unet_keys = [k for k in all_keys if k.startswith("unet")]
text_encoder_keys = [k for k in all_keys if k.startswith("text_encoder")]
text_model_keys = [k for k in all_keys if k.startswith("text_model")]

print(f"权重统计:")
print(f"   - UNet 相关 Key 数量: {len(unet_keys)}")
print(f"   - Text Encoder 相关 Key 数量: {len(text_encoder_keys)}")
print(f"   - Text Model (无前缀) 相关 Key 数量: {len(text_model_keys)}")

print("\n--- Text Encoder / Model Key 样例 (前 10 个) ---")
te_samples = text_encoder_keys if text_encoder_keys else text_model_keys
if te_samples:
    for k in te_samples[:10]:
        print(k)
else:
    print("❌ Warning: No Text Encoder related weights found in the .bin file!")
    print("This indicates that your training script did not successfully extract Text Encoder parameters during save_checkpoint.")