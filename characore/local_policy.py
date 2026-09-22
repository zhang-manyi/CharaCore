"""Offline Transformers inference; reuses the existing local DPO model loader only."""
import importlib.metadata
import os
from pathlib import Path

from characore.protocol import digest


class LocalPolicy:
    def __init__(self, base, device="cuda", quantize=False, max_new_tokens=192, seed=17):
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        base = Path(base).resolve()
        if not base.is_dir():
            raise ValueError("--base must be an existing local model directory")
        if not 1 <= max_new_tokens <= 1024:
            raise ValueError("max_new_tokens must be 1..1024")
        import torch
        from transformers import set_seed
        from scripts.train_dpo import load_model

        torch.set_num_threads(4)
        set_seed(seed)
        self.model, self.tokenizer = load_model(str(base), device=device, quantize=quantize)
        self.model.eval()
        self.model.config.use_cache = True
        self.max_new_tokens = max_new_tokens
        self.metadata = dict(kind="local_transformers", claim="真实本地基座；设计开发任务单次运行，非盲测。",
                             base=str(base), device=device, quantized_4bit=quantize,
                             dtype=str(self.model.dtype), seed=seed, do_sample=False,
                             max_new_tokens=max_new_tokens, enable_thinking=False,
                             packages={p: importlib.metadata.version(p) for p in ("torch", "transformers", "accelerate")},
                             model_files_sha256={p.name: digest(p) for p in sorted(base.iterdir()) if p.is_file()},
                             source_sha256={"local_policy.py": digest(__file__),
                                            "train_dpo.py": digest(Path(__file__).resolve().parents[1] / "scripts/train_dpo.py")},
                             device_name=torch.cuda.get_device_name(0) if device == "cuda" else "cpu")

    def __call__(self, messages):
        import torch

        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False,
                                                    add_generation_prompt=True, enable_thinking=False)
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        count = inputs["input_ids"].shape[1]
        if count + self.max_new_tokens > self.model.config.max_position_embeddings:
            raise ValueError("context budget exceeded; refusing silent memory truncation")
        with torch.inference_mode():
            generated = self.model.generate(**inputs, do_sample=False, max_new_tokens=self.max_new_tokens,
                                            pad_token_id=self.tokenizer.pad_token_id,
                                            eos_token_id=self.tokenizer.eos_token_id)
        ids = generated[0, count:].tolist()
        raw = self.tokenizer.decode(ids, skip_special_tokens=True)
        return raw, dict(prompt_text=prompt, input_tokens=count, output_tokens=len(ids),
                         output_token_ids=ids, reached_token_limit=len(ids) >= self.max_new_tokens,
                         cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated() if self.model.device.type == "cuda" else None)
