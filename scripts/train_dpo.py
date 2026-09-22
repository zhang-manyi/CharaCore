"""Offline TRL LoRA DPO with update/reload evidence. Never resumes/overwrites a run."""
import argparse
from contextlib import nullcontext
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_model(base, tiny=False, quantize=False, device="cpu", precision=None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    cuda = torch.device(device).type == "cuda"
    if precision is None:
        dtype = torch.bfloat16 if cuda else torch.float32  # Preserve the existing DPO configuration.
    else:
        from characore.precision import select_precision
        chosen = select_precision(device, precision)
        dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[chosen]
    if tiny:
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
        vocab = {s: i for i, s in enumerate("<pad> <eos> <unk> medic guard help patient bridge safe inspect abandon harm now please calm act".split())}
        t = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
        t.pre_tokenizer = Whitespace()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=t, pad_token="<pad>", eos_token="<eos>", unk_token="<unk>")
        model = GPT2LMHeadModel(GPT2Config(vocab_size=len(vocab), n_positions=128, n_embd=32,
                    n_layer=2, n_head=2, bos_token_id=1, eos_token_id=1, pad_token_id=0,
                    resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0))
    else:
        tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
        options = {"local_files_only": True, "torch_dtype": dtype,
                   "attn_implementation": "sdpa"}
        if quantize:
            if not cuda:
                raise ValueError("4-bit experiment requires CUDA")
            options["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True,
                    bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=dtype)
            options["device_map"] = {"": torch.device(device).index or 0}
        model = AutoModelForCausalLM.from_pretrained(base, **options)
    if not quantize:
        model.to(device)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "left"
    model.config.use_cache = False
    return model, tokenizer


def run(args):
    import torch
    from datasets import Dataset
    from peft import LoraConfig, PeftModel
    from transformers import set_seed
    from trl import DPOConfig, DPOTrainer
    from characore.protocol import load_suite, dpo_records, digest

    torch.set_num_threads(4)
    set_seed(args.seed)
    model, tokenizer = load_model(args.base, args.tiny, args.quantize, args.device)
    if args.tiny:
        # Random tiny network is a mechanics test, never a capability result.
        model.save_pretrained(args.output / "tiny_base")
        tokenizer.save_pretrained(args.output / "tiny_base")
        base_path = str((args.output / "tiny_base").resolve())
        records = [{"prompt": "medic help patient ", "chosen": "help patient now", "rejected": "abandon patient now"},
                   {"prompt": "guard bridge ", "chosen": "inspect bridge safe", "rejected": "ignore harm now"}]
        target_modules = ["c_attn", "c_proj"]
    else:
        profiles, train, _ = load_suite(args.suite)
        records = dpo_records(train, profiles, tokenizer)
        if not records:
            raise ValueError("Selected suite contains no strict preference pairs")
        base_path = str(Path(args.base).resolve())
        target_modules = ["q_proj", "v_proj"]
    token_audit = []
    for row in records:
        lengths = {key: len(tokenizer(row[key], add_special_tokens=False)["input_ids"]) for key in row}
        if lengths["prompt"] + max(lengths["chosen"], lengths["rejected"]) + 1 > args.max_length:
            raise ValueError(f"Refusing silent DPO truncation: {lengths}, budget={args.max_length}")
        token_audit.append(lengths)
    manifest = {"status": "running", "claim": "training mechanics only; capability improvement unverified",
        "base": base_path, "tiny_random_model": args.tiny, "quantized_4bit": args.quantize,
        "seed": args.seed, "steps": args.steps, "beta": 0.1, "learning_rate": args.learning_rate,
        "max_length": args.max_length, "reference": "same frozen base with LoRA disabled",
        "training_records": len(records), "token_audit": token_audit,
        "suite": str(args.suite.resolve()) if not args.tiny else None,
        "suite_hashes": {p: digest(args.suite / p) for p in ["profiles.json", "train.json", "eval.json", "freeze.json", "train_manifest.json", "evaluation_plan.json"]} if not args.tiny else {},
        "packages": {p: importlib.metadata.version(p) for p in ["torch", "transformers", "peft", "trl", "datasets", "accelerate"]},
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "evaluation_plan_sha256": digest(args.suite / "evaluation_plan.json") if not args.tiny else None,
        "base_files_sha256": {p.name: digest(p) for p in Path(base_path).iterdir()
                              if p.suffix in {".json", ".safetensors"}}}
    write_json(args.output / "manifest.json", manifest)
    cfg = DPOConfig(output_dir=str(args.output / "trainer"), max_steps=args.steps,
        per_device_train_batch_size=1, gradient_accumulation_steps=1,
        learning_rate=args.learning_rate, beta=0.1, max_length=args.max_length,
        max_prompt_length=None, max_completion_length=None, logging_steps=1,
        save_strategy="no", eval_strategy="no", report_to="none", seed=args.seed,
        data_seed=args.seed, bf16=args.device == "cuda", fp16=False,
        use_cpu=args.device == "cpu", gradient_checkpointing=not args.tiny,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_pin_memory=False, remove_unused_columns=False,
        dataset_num_proc=None, disable_tqdm=True)
    trainer = DPOTrainer(model=model, ref_model=None, args=cfg,
        processing_class=tokenizer, train_dataset=Dataset.from_list(records),
        peft_config=LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0,
                               target_modules=target_modules, task_type="CAUSAL_LM"))
    policy = trainer.model
    trainables = {n: p for n, p in policy.named_parameters() if p.requires_grad}
    if not trainables or any("lora_" not in n for n in trainables):
        raise AssertionError("Expected LoRA-only trainable parameters")
    before = {n: p.detach().cpu().clone() for n, p in trainables.items()}
    grads = {"calls": 0, "nonzero_calls": 0, "all_finite": True, "max_abs": 0.0}
    def observe(grad):
        grads["calls"] += 1
        grads["nonzero_calls"] += int(torch.count_nonzero(grad).item() > 0)
        grads["all_finite"] &= bool(torch.isfinite(grad).all())
        grads["max_abs"] = max(grads["max_abs"], grad.detach().float().abs().max().item())
    hooks = [p.register_hook(observe) for p in trainables.values()]
    probe = tokenizer(records[0]["prompt"] + records[0]["chosen"], return_tensors="pt").to(policy.device)
    # Short probe avoids allocating long vocabulary logits on a 6 GB GPU.
    probe = {k: v[:, -16:] for k, v in probe.items() if k in {"input_ids", "attention_mask"}}
    def logits():
        policy.eval()
        amp = torch.autocast("cuda", dtype=torch.bfloat16) if args.device == "cuda" else nullcontext()
        with torch.no_grad(), amp:
            return policy(**probe).logits.detach().float().cpu()
    initial = logits()
    with policy.disable_adapter():
        reference_before = logits()
    result = trainer.train()
    for hook in hooks:
        hook.remove()
    trained = logits()
    with policy.disable_adapter():
        reference_after = logits()
    changes = {n: (p.detach().cpu().float() - before[n].float()).abs().max().item() for n, p in trainables.items()}
    b_nonzero = sum(torch.count_nonzero(p).item() for n, p in trainables.items() if "lora_B" in n)
    if not grads["all_finite"] or not grads["nonzero_calls"] or not any(changes.values()) or not b_nonzero:
        raise AssertionError("No finite nonzero gradient/update evidence")
    if not torch.equal(reference_before, reference_after):
        raise AssertionError("Frozen reference logits changed")
    if torch.equal(initial, trained):
        raise AssertionError("Adapter changed but probe logits did not")
    policy.save_pretrained(args.output / "adapter")
    tokenizer.save_pretrained(args.output / "adapter")
    # Keep the frozen base in memory; unload old adapter, load from disk.
    frozen_base = policy.unload()
    policy = PeftModel.from_pretrained(frozen_base, args.output / "adapter", is_trainable=False)
    restored = logits()
    reload_error = (trained - restored).abs().max().item()
    if not torch.allclose(trained, restored, atol=1e-5, rtol=1e-5):
        raise AssertionError(f"Saved/reloaded logits differ: {reload_error}")
    proof = {"gradient": grads, "changed_tensors": sum(v > 0 for v in changes.values()),
        "parameter_max_deltas": changes, "nonzero_lora_B_elements": b_nonzero,
        "probe_logit_max_delta": (trained - initial).abs().max().item(),
        "reference_logits_unchanged": True, "reload_logit_max_error": reload_error,
        "optimizer_steps": trainer.state.global_step, "train_metrics": result.metrics,
        "training_history": trainer.state.log_history,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated() if args.device == "cuda" else None}
    write_json(args.output / "verification.json", proof)
    manifest["status"] = "mechanics_verified"
    manifest["adapter_sha256"] = digest(args.output / "adapter/adapter_model.safetensors")
    write_json(args.output / "manifest.json", manifest)
    print(json.dumps({k: proof[k] for k in ["changed_tensors", "nonzero_lora_B_elements", "probe_logit_max_delta", "reload_logit_max_error", "optimizer_steps"]}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--base", help="Local pretrained model directory; required unless --tiny")
    parser.add_argument("--suite", type=Path, help="Versioned data directory; required unless --tiny")
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if not args.tiny and (not args.base or not args.suite):
        parser.error("--base and --suite are required unless --tiny")
    if args.tiny and args.suite:
        parser.error("--tiny uses synthetic mechanics data; omit --suite")
    if args.steps <= 0 or args.max_length <= 0:
        parser.error("steps and max-length must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        run(args)
    except Exception:
        (args.output / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
