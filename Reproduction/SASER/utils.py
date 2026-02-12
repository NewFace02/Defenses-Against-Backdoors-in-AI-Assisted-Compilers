import gc
import torch
import numpy as np

# 清理显存
def clear_gpu():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

# 设置随机种子
def set_seed(seed=2026):
    torch.manual_seed(seed)
    np.random.seed(seed)

def get_random_bits(length, n):
    rand = np.random.randint(0, 2, size=length).tolist()
    pad = (n - (len(rand) % n)) % n
    return rand + [0] * pad
