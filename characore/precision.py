"""Select native CUDA precision; V100 uses FP16 even if BF16 emulation is available."""


def select_precision(device, requested="auto", capability=None):
    if requested not in ("auto", "fp16", "bf16", "fp32"):
        raise ValueError("precision must be auto/fp16/bf16/fp32")
    cuda = str(device).startswith("cuda")
    if not cuda:
        if requested not in ("auto", "fp32"):
            raise ValueError("CPU path requires fp32")
        return "fp32"
    if capability is None:
        import torch
        capability = torch.cuda.get_device_capability(device)
    native_bf16 = capability[0] >= 8
    if requested == "bf16" and not native_bf16:
        raise ValueError("this GPU lacks native BF16; select fp16 (V100) or auto")
    return ("bf16" if native_bf16 else "fp16") if requested == "auto" else requested
