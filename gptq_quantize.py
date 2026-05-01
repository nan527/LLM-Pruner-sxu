import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import datasets
import torch
import time
from LLMPruner.peft import PeftModel
from LLMPruner.evaluator.ppl import PPLMetric

device = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# Weight-only quantization functions (simulates GPTQ/AWQ)
# ============================================================
def quantize_weight_per_group(weight, bits=4, group_size=128):
    """Per-group weight-only quantization to INT4/INT8"""
    orig_shape = weight.shape
    orig_dtype = weight.dtype
    w = weight.float()

    # Reshape for group quantization
    if w.shape[-1] % group_size != 0:
        # Pad if needed
        pad_size = group_size - (w.shape[-1] % group_size)
        w = torch.nn.functional.pad(w, (0, pad_size))
    
    w_grouped = w.reshape(-1, group_size)
    
    # Compute scales and zeros per group
    w_min = w_grouped.min(dim=1, keepdim=True).values
    w_max = w_grouped.max(dim=1, keepdim=True).values
    
    qmax = 2 ** bits - 1
    scales = (w_max - w_min) / qmax
    scales = scales.clamp(min=1e-8)
    zeros = w_min
    
    # Quantize and dequantize (simulate)
    w_quant = torch.round((w_grouped - zeros) / scales).clamp(0, qmax)
    w_deq = w_quant * scales + zeros
    
    # Reshape back
    w_deq = w_deq.reshape(w.shape)
    if w_deq.shape != orig_shape:
        w_deq = w_deq[:orig_shape[0], :orig_shape[1]]
    
    return w_deq.to(orig_dtype)


def quantize_model_weights(model, bits=4, group_size=128):
    """Apply weight-only quantization to all Linear layers"""
    count = 0
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            module.weight.data = quantize_weight_per_group(
                module.weight.data, bits=bits, group_size=group_size
            )
            count += 1
    print(f"    Quantized {count} Linear layers to INT{bits} (group_size={group_size})")
    return model


def calc_model_size_bits(model, quant_bits=4, group_size=128):
    """Estimate model size after weight-only quantization"""
    total_bits = 0
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            numel = module.weight.numel()
            n_groups = (numel + group_size - 1) // group_size
            # Quantized weights + scales + zeros (FP16 each)
            total_bits += numel * quant_bits + n_groups * 16 * 2
        elif hasattr(module, 'weight') and module.weight is not None:
            if not any(isinstance(m, torch.nn.Linear) for m in module.children()):
                total_bits += module.weight.numel() * module.weight.element_size() * 8
    return total_bits / 8 / 1024 / 1024  # MB


# ============================================================
# Load pruned + LoRA model
# ============================================================
print("Loading pruned model + LoRA...")
pruned_dict = torch.load(
    "prune_log/tinyllama_prune/pytorch_model.bin",
    map_location='cpu'
)
tokenizer = pruned_dict['tokenizer']
model = pruned_dict['model']
model.half().to(device)

model = PeftModel.from_pretrained(
    model,
    "tune_log/tinyllama_tune_25",
    torch_dtype=torch.float16,
)
model = model.merge_and_unload()
model.to(device)

# ============================================================
# INT4 Weight-Only Quantization (GPTQ-style)
# ============================================================
print("\n[1] INT4 weight-only quantization (GPTQ-style)...")
est_size_int4 = calc_model_size_bits(model, quant_bits=4, group_size=128)
print(f"    Estimated INT4 model size: {est_size_int4:.1f} MB")

model_int4 = quantize_model_weights(model, bits=4, group_size=128)

start = time.time()
ppl_int4 = PPLMetric(model_int4, tokenizer, ['wikitext2'], seq_len=128, batch_size=1, device=device)
time_int4 = time.time() - start
print(f"    PPL: {ppl_int4['wikitext2']:.2f}, Time: {time_int4:.1f}s")

# ============================================================
# INT8 Weight-Only Quantization
# ============================================================
print("\n[2] INT8 weight-only quantization...")
# Reload model for INT8 (since INT4 modified weights in-place)
del model_int4
torch.cuda.empty_cache()

pruned_dict = torch.load(
    "prune_log/tinyllama_prune/pytorch_model.bin",
    map_location='cpu'
)
model = pruned_dict['model']
model.half().to(device)
model = PeftModel.from_pretrained(
    model,
    "tune_log/tinyllama_tune_25",
    torch_dtype=torch.float16,
)
model = model.merge_and_unload()
model.to(device)

est_size_int8 = calc_model_size_bits(model, quant_bits=8, group_size=128)
print(f"    Estimated INT8 model size: {est_size_int8:.1f} MB")

model_int8 = quantize_model_weights(model, bits=8, group_size=128)

start = time.time()
ppl_int8 = PPLMetric(model_int8, tokenizer, ['wikitext2'], seq_len=128, batch_size=1, device=device)
time_int8 = time.time() - start
print(f"    PPL: {ppl_int8['wikitext2']:.2f}, Time: {time_int8:.1f}s")

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 70)
print("FULL PIPELINE SUMMARY")
print("=" * 70)
print(f"{'Model Version':<35} {'PPL':>8} {'Size(MB)':>10} {'Time':>10}")
print("-" * 70)
print(f"{'Original TinyLlama (FP16)':<35} {'17.42':>8} {'~2200':>10} {'--':>10}")
print(f"{'Pruned 25%':<35} {'35.95':>8} {'~1762':>10} {'--':>10}")
print(f"{'Pruned 25% + LoRA (FP16)':<35} {'26.71':>8} {'~1762':>10} {'--':>10}")
print(f"{'Pruned + LoRA + W8 (per-group)':<35} {ppl_int8['wikitext2']:>8.2f} {f'~{est_size_int8:.0f}':>10} {f'{time_int8:.0f}s':>10}")
print(f"{'Pruned + LoRA + W4 (per-group)':<35} {ppl_int4['wikitext2']:>8.2f} {f'~{est_size_int4:.0f}':>10} {f'{time_int4:.0f}s':>10}")
print(f"{'Pruned + LoRA + Dynamic INT8':<35} {'81.82':>8} {'~250':>10} {'589s':>10}")
print("=" * 70)
print("\nNote: W4/W8 = weight-only quantization (similar to GPTQ/AWQ)")
print("      Dynamic INT8 = PyTorch dynamic quantization (activations also quantized)")
