"""Batched reply generation shared by base-reply freezing, the survey and evaluation."""
from characore.persona import render_prompt
from characore.style_rewards import clean_reply


def generate(model, tokenizer, rows, max_new_tokens, batch_size=8, generation=None):
    """Greedy by default. With a sampling GenerationConfig, returns num_return_sequences replies per row.

    Returns {row id: [raw decoded text, ...]}; raw text is kept, clean_reply is applied by callers.
    """
    import torch
    from transformers import GenerationConfig

    if generation is None:
        generation = GenerationConfig(do_sample=False, max_new_tokens=max_new_tokens,
                                      pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    per_row = generation.num_return_sequences or 1
    limit = model.config.max_position_embeddings
    out = {}
    model.eval()
    for start in range(0, len(rows), batch_size):
        chunk = rows[start:start + batch_size]
        prompts = [render_prompt(tokenizer, r, max_new_tokens, limit) for r in chunk]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        width = inputs["input_ids"].shape[1]  # left padding: every completion starts here
        with torch.inference_mode():
            ids = model.generate(**inputs, generation_config=generation)
        for k, row in enumerate(chunk):
            out[row["id"]] = [tokenizer.decode(ids[k * per_row + j, width:], skip_special_tokens=True)
                              for j in range(per_row)]
    return out


def first_clean(outputs):
    return {key: clean_reply(values[0]) for key, values in outputs.items()}
