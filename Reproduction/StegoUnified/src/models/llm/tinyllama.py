from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


def load_tinyllama(model_source="TinyLlama/TinyLlama-1.1B-Chat-v1.0", device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("[+] Loading model...")
    tok = AutoTokenizer.from_pretrained(model_source)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        torch_dtype=torch.float16
    ).to(device)

    return model, tok