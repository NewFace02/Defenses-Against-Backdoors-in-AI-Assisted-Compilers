import torch
import math
import numpy as np
from src.quantizer import BlockQuantizer
from src.attack.protocol import StegoProtocol, TRIGGER_SEQ

class SASERCore:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
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
            inputs = self.tokenizer(q + " " + a, return_tensors="pt").to(self.device)
            n_tokens = inputs["input_ids"].numel()
            out = self.model(**inputs, labels=inputs["input_ids"])
            loss = out.loss.item()
            if math.isnan(loss) or math.isinf(loss): return float('inf'), 0.0
            nll_sum += loss * n_tokens
            total_tokens += n_tokens
            gen_out = self.model.generate(**self.tokenizer(q, return_tensors="pt").to(self.device), 
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
