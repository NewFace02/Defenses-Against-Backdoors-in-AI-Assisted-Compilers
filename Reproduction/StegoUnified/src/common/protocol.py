from collections import Counter

# 16-bit 触发器序列 (用于定位 Payload)
TRIGGER_SEQ = [1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0]

class StegoProtocol:
    """
    通信协议：ECC 与 封装
    实现 5x 重复编码，增强抗噪能力
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
