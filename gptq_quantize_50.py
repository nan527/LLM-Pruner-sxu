import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import datasets
import torch
import time
from LLMPruner.peft import PeftModel
from LLMPruner.evaluator.ppl import PPLMetric

device = "cuda" if torch.cuda.is_available() else "cpu"

def quantize_weight_per_group(weight, bits=4, group_size=128):
    orig_shape = weight.shape
    orig_dtype = weight.dtype
    w = weight.float()
    if w.shape[-1] % group_size != 0:
        pad_size = group_size - (w.shape[-1] % group_size)
        w = torch.nn.functional.pad(w, (0, pad_size))
    w_grouped = w.reshape(-1, group_size)
    w_min = w_grouped.min(dim=1, keepdim=True).values
    w_max = w_grouped.max(dim=1, keepdim=True).values
    qmax = 2 ** bits - 1
    scales = (w_max - w_min) / qmax
    scales = scales.clamp(min=1e-8)
    zeros = w_min
    w_quant = torch.round((w_grouped - zeros) / scales).clamp(0, qmax)
    w_deq = w_quant * scales + zeros
    w_deq = w_deq.reshape(w.shape)
    if w_deq.shape != orig_shape:
        w_deq = w_deq[:orig_shape[0], :orig_shape[1]]
    return w_deq.to(orig_dtype)

def quantize_model_weights(model, bits=4, group_size=128):
    count = 0
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            module.weight.data = quantize_weight_per_group(module.weight.data, bits=bits, group_size=group_size)
            count += 1
    print(f"    Quantized {count} Linear layers to INT{bits}")
    return model

def calc_model_size_bits(model, quant_bits=4, group_size=128):
    total_bits = 0
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            numel = module.weight.numel()
            n_groups = (numel + group_size - 1) // group_size
            total_bits += numel * quant_bits + n_groups * 16 * 2
        elif hasattr(module, 'weight') and module.weight is not None:
            if not any(isinstance(m, torch.nn.Linear) for m in module.children()):
                total_bits += module.weight.numel() * module.weight.element_size() * 8
    return total_bits / 8 / 1024 / 1024

def load_model_with_lora(prune_path, lora_path):
    pruned_dict = torch.load(prune_path, map_location='cpu')
    tokenizer = pruned_dict['tokenizer']
    model = pruned_dict['model']
    model.half().to(device)
    model = PeftModel.from_pretrained(model, lora_path, torch_dtype=torch.float16)
    model = model.merge_and_unload()
    model.to(device)
    return model, tokenizer

# ============================================================
# 50% pruned + LoRA + W4
# ============================================================
print("=== 50% Pruned + LoRA + W4 ===")
model, tokenizer = load_model_with_lora(
    "prune_log/tinyllama_prune_50/pytorch_model.bin",
    "tune_log/tinyllama_tune_50"
)
est_size_w4_50 = calc_model_size_bits(model, quant_bits=4, group_size=128)
model = quantize_model_weights(model, bits=4, group_size=128)
start = time.time()
ppl_w4_50 = PPLMetric(model, tokenizer, ['wikitext2'], seq_len=128, batch_size=1, device=device)
time_w4_50 = time.time() - start
print(f"    PPL: {ppl_w4_50['wikitext2']:.2f}, Size: ~{est_size_w4_50:.0f} MB, Time: {time_w4_50:.0f}s")

del model
torch.cuda.empty_cache()

# ============================================================
# 50% pruned + LoRA + W8
# ============================================================
print("\n=== 50% Pruned + LoRA + W8 ===")
model, tokenizer = load_model_with_lora(
    "prune_log/tinyllama_prune_50/pytorch_model.bin",
    "tune_log/tinyllama_tune_50"
)
est_size_w8_50 = calc_model_size_bits(model, quant_bits=8, group_size=128)
model = quantize_model_weights(model, bits=8, group_size=128)
start = time.time()
ppl_w8_50 = PPLMetric(model, tokenizer, ['wikitext2'], seq_len=128, batch_size=1, device=device)
time_w8_50 = time.time() - start
print(f"    PPL: {ppl_w8_50['wikitext2']:.2f}, Size: ~{est_size_w8_50:.0f} MB, Time: {time_w8_50:.0f}s")

# ============================================================
# Full Summary
# ============================================================
print("\n" + "=" * 75)
print("COMPLETE COMPARISON TABLE")
print("=" * 75)
print(f"{'Model Version':<40} {'PPL':>8} {'Size(MB)':>10} {'Ratio':>8}")
print("-" * 75)
print(f"{'Original TinyLlama (FP16)':<40} {'17.42':>8} {'2200':>10} {'1.0x':>8}")
print()
print(f"{'--- Pruning 25% ---':<40}")
print(f"{'  Pruned 25%':<40} {'35.95':>8} {'1762':>10} {'1.2x':>8}")
print(f"{'  + LoRA (FP16)':<40} {'26.71':>8} {'1762':>10} {'1.2x':>8}")
print(f"{'  + LoRA + W8':<40} {'26.71':>8} {'969':>10} {'2.3x':>8}")
print(f"{'  + LoRA + W4':<40} {'30.75':>8} {'560':>10} {'3.9x':>8}")
print()
print(f"{'--- Pruning 50% ---':<40}")
print(f"{'  Pruned 50%':<40} {'90.72':>8} {'1426':>10} {'1.5x':>8}")
print(f"{'  + LoRA (FP16)':<40} {'45.18':>8} {'1426':>10} {'1.5x':>8}")
print(f"{'  + LoRA + W8':<40} {ppl_w8_50['wikitext2']:>8.2f} {f'{est_size_w8_50:.0f}':>10} {f'{2200/est_size_w8_50:.1f}x':>8}")
print(f"{'  + LoRA + W4':<40} {ppl_w4_50['wikitext2']:>8.2f} {f'{est_size_w4_50:.0f}':>10} {f'{2200/est_size_w4_50:.1f}x':>8}")
print("=" * 75)
