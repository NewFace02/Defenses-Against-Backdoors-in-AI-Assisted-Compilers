import torch

class BlockQuantizer:
    """
    GGUF 量化
    模拟GGUF格式的Q4_0/Q8_0分块对称量化机制。
    """
    @staticmethod
    def quantize(w, bits, block_size=32):
        orig_shape = w.shape
        target_device = w.device
        # 使用 float64 进行高精度 Scale 计算
        w_flat = w.view(-1).to(torch.float64)
        
        # 补齐 Padding，确保能被 block_size 整除
        pad_len = (block_size - (w_flat.numel() % block_size)) % block_size
        if pad_len > 0: w_flat = torch.nn.functional.pad(w_flat, (0, pad_len))
        
        blocks = w_flat.view(-1, block_size)
        
        # GGUF 对称量化公式: Scale = max(|w|) / Q_max
        max_abs = blocks.abs().max(dim=1, keepdim=True)[0]
        q_max = 2**(bits-1) - 1
        scale = max_abs / q_max
        scale[scale == 0] = 1.0 # 防止除零
        
        # 量化: q = round(w / scale)
        q_blocks = torch.round(blocks / scale).clamp(-q_max, q_max).to(torch.int8)
        return q_blocks, scale.to(target_device), orig_shape, pad_len

    @staticmethod
    def dequantize(q_blocks, scale, orig_shape, pad_len):
        target_device = scale.device
        # 反量化: w = q * scale
        w_recon = (q_blocks.to(torch.float64) * scale).view(-1)
        if pad_len > 0: w_recon = w_recon[:-pad_len]
        return w_recon.view(orig_shape).to(torch.float16).to(target_device)

    # Scale Locking 保护掩码
    @staticmethod
    def get_protection_mask(w, block_size=32):
        """
        计算保护掩码：找出所有决定 Scale 的最大值位置，禁止修改
        """
        target_device = w.device
        w_flat = w.view(-1).to(torch.float64)
        pad_len = (block_size - (w_flat.numel() % block_size)) % block_size
        if pad_len > 0: w_flat = torch.nn.functional.pad(w_flat, (0, pad_len))
        
        blocks = w_flat.view(-1, block_size)
        
        # 找出每个 block 的最大绝对值
        max_vals = blocks.abs().max(dim=1, keepdim=True)[0]
        
        # 锁定所有等于最大值的位置 
        local_mask = (blocks.abs() == max_vals)
        
        # True = 可写入 (非最大值), False = 保护 (最大值)
        mask = (~local_mask).flatten()
        
        if pad_len > 0: mask = mask[:-pad_len]
        return mask
