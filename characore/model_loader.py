"""Strictly offline local model loading. Never downloads; never auto-truncates."""
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Vocabulary for the random tiny model: mechanics checks only, never a capability claim.
TINY_VOCAB = "<pad> <eos> <unk> medic guard help patient bridge safe inspect abandon harm now please calm act".split()


def load_model(base, tiny=False, quantize=False, device="cpu", precision="auto"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from characore.precision import select_precision

    cuda = torch.device(device).type == "cuda"
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16,
             "fp32": torch.float32}[select_precision(device, precision)]
    if tiny:
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
        t = Tokenizer(WordLevel({s: i for i, s in enumerate(TINY_VOCAB)}, unk_token="<unk>"))
        t.pre_tokenizer = Whitespace()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=t, pad_token="<pad>",
                                            eos_token="<eos>", unk_token="<unk>")
        model = GPT2LMHeadModel(GPT2Config(vocab_size=len(TINY_VOCAB), n_positions=128,
                    n_embd=32, n_layer=2, n_head=2, bos_token_id=1, eos_token_id=1,
                    pad_token_id=0, resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0))
    else:
        tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
        options = {"local_files_only": True, "torch_dtype": dtype, "attn_implementation": "sdpa"}
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
