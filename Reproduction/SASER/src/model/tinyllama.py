from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def load_tinyllama(device):
    print("[+] Loading model to GPU directly...")
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
    return model, tok
