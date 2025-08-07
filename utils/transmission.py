import tenseal as ts
import pickle
import zlib
import random
import numpy as np
from pymceliece import McEliece
from typing import Optional, Any, Tuple


class PrivacyTransmission:
    def __init__(self,
                 data: Any,
                 context: Optional[ts.Context] = None,
                 # 同态加密参数 (BFV方案)
                 he_poly_modulus_degree: int = 4096,
                 he_plain_modulus: int = 1032193,
                 he_generate_galois: bool = True,
                 # McEliece参数
                 mc_n: int = 1632,
                 mc_k: int = 1269,
                 mc_t: int = 11,
                 # 处理参数
                 redundancy_ratio: float = 0.2,
                 compress_level: int = 6,
                 serialize_protocol: int = 5,
                 # 噪声参数
                 noise_probability: float = 0.01,  # 字节被篡改的概率
                 max_noise_intensity: int = 3):  # 最大字节篡改强度
        """
        初始化隐私保护处理实例，模拟数据加密、处理和噪声干扰过程
        不进行实际文件传输，仅在内存中完成整个流程
        """
        self.data = data
        self.original_data = pickle.dumps(data, protocol=serialize_protocol)  # 保存原始数据用于校验
        self.processed_data = None  # 处理后的最终数据
        self.encoded_bytes = None  # 序列化后的原始字节
        self.he_encrypted = None  # 同态加密后的数据
        self.serialized = None  # McEliece编码后的数据
        self.compressed = None  # 压缩后的数据
        self.noisy_data = None  # 加入噪声后的数据

        # 存储配置参数
        self.he_poly_modulus_degree = he_poly_modulus_degree
        self.he_plain_modulus = he_plain_modulus
        self.he_generate_galois = he_generate_galois
        self.mc_n = mc_n
        self.mc_k = mc_k
        self.mc_t = mc_t
        self.redundancy_ratio = redundancy_ratio
        self.compress_level = compress_level
        self.serialize_protocol = serialize_protocol
        self.noise_probability = noise_probability
        self.max_noise_intensity = max_noise_intensity

        # 初始化同态加密上下文
        self.context = context if context else self._create_he_context()

        # 初始化McEliece密码体制
        self.mc = McEliece(n=mc_n, k=mc_k, t=mc_t)
        self.pub_key, self.priv_key = self.mc.generate_keypair()

        # 存储评估指标
        self.he_accuracy = None  # 同态加密准确率
        self.error_correction_rate = None  # 纠错成功率
        self.total_errors = 0  # 总错误数

    def _create_he_context(self) -> ts.Context:
        """根据配置创建同态加密上下文"""
        context = ts.context(
            ts.SCHEME_TYPE.BFV,
            poly_modulus_degree=self.he_poly_modulus_degree,
            plain_modulus=self.he_plain_modulus
        )

        if self.he_generate_galois:
            context.generate_galois_keys()

        return context

    def _encode(self) -> bytes:
        """序列化数据为字节流"""
        self.encoded_bytes = pickle.dumps(
            self.data,
            protocol=self.serialize_protocol
        )
        return self.encoded_bytes

    def _decode(self) -> Any:
        """从字节流反序列化数据"""
        if not self.encoded_bytes:
            raise ValueError("无可用解码数据，请先执行解密步骤")

        self.processed_data = pickle.loads(self.encoded_bytes)
        return self.processed_data

    def _he_encrypt(self) -> ts.BFVVector:
        """使用同态加密加密数据"""
        if not self.encoded_bytes:
            self._encode()

        int_data = [int(byte) for byte in self.encoded_bytes]
        self.he_encrypted = ts.bfv_encrypt(self.context, int_data)
        return self.he_encrypted

    def _he_decrypt(self) -> bytes:
        """解密同态加密数据"""
        if not self.he_encrypted:
            raise ValueError("无可用加密数据，请先执行加密步骤")

        decrypted_ints = self.he_encrypted.decrypt()
        self.encoded_bytes = bytes(decrypted_ints)
        return self.encoded_bytes

    def _serialize(self) -> bytes:
        """使用McEliece进行带纠错的序列化"""
        if not self.he_encrypted:
            self._he_encrypt()

        he_bytes = self.he_encrypted.serialize()
        redundancy_length = int(len(he_bytes) * self.redundancy_ratio)
        total_length = len(he_bytes) + redundancy_length
        self.serialized = self.mc.encrypt(he_bytes, self.pub_key, total_length)
        return self.serialized

    def _deserialize(self) -> ts.BFVVector:
        """使用McEliece进行解码和纠错"""
        if not self.serialized:
            raise ValueError("无可用序列化数据，请先执行序列化步骤")

        he_bytes = self.mc.decrypt(self.serialized, self.priv_key)
        self.he_encrypted = ts.lazy_bfv_encrypt(self.context, he_bytes)
        return self.he_encrypted

    def _compress(self) -> bytes:
        """压缩数据"""
        if not self.serialized:
            self._serialize()

        self.compressed = zlib.compress(
            self.serialized,
            level=self.compress_level
        )
        return self.compressed

    def _decompress(self) -> bytes:
        """解压缩数据"""
        if not self.compressed:
            raise ValueError("无可用压缩数据，请先执行压缩步骤")

        self.serialized = zlib.decompress(self.compressed)
        return self.serialized

    def _add_transmission_noise(self, data: bytes, noise_prob: float = None) -> Tuple[bytes, int]:
        """
        向数据添加噪声，模拟网络传输错误
        返回带噪声的数据和错误计数
        """
        noise_prob = noise_prob if noise_prob is not None else self.noise_probability
        data_list = list(data)
        corrupted_count = 0

        for i in range(len(data_list)):
            if random.random() < noise_prob:
                noise = random.randint(1, self.max_noise_intensity)
                data_list[i] = (data_list[i] ^ noise) % 256
                corrupted_count += 1

        if corrupted_count > 0:
            print(f"模拟传输中产生了 {corrupted_count} 处数据错误")

        return bytes(data_list), corrupted_count

    def process(self, use_noise: bool = True, noise_prob: Optional[float] = None) -> Tuple[Any, bool]:
        """
        处理数据的完整流程：编码→加密→序列化→压缩→加噪声→解压缩→反序列化→解密→解码

        Args:
            use_noise: 是否添加噪声
            noise_prob: 自定义噪声概率，None则使用实例化时的默认值

        Returns:
            处理后的数据和处理成功状态
        """
        try:
            # 正向处理流程
            self._encode()
            self._he_encrypt()
            self._serialize()
            self._compress()

            # 根据参数决定是否添加噪声
            if use_noise:
                applied_noise_prob = noise_prob if noise_prob is not None else self.noise_probability
                self.noisy_data, self.total_errors = self._add_transmission_noise(self.compressed, applied_noise_prob)
            else:
                self.noisy_data = self.compressed
                self.total_errors = 0

            # 反向处理流程
            self.compressed = self.noisy_data  # 使用带噪声的数据进行还原
            self._decompress()
            self._deserialize()  # McEliece纠错
            self._he_decrypt()
            self._decode()

            print("数据处理流程完成")
            return self.processed_data, True

        except Exception as e:
            print(f"数据处理出错: {str(e)}")
            return None, False

    def evaluate_he_performance(self) -> float:
        """
        评估同态加密效果：在无噪声情况下比较原始数据与处理后数据的一致性
        """
        # 执行无噪声处理
        self.process(use_noise=False)

        # 计算准确率
        processed_bytes = pickle.dumps(self.processed_data, protocol=self.serialize_protocol)
        match_count = sum(1 for o, d in zip(self.original_data, processed_bytes) if o == d)
        self.he_accuracy = match_count / len(self.original_data) * 100

        print(f"同态加密效果评估: {self.he_accuracy:.2f}% 字节匹配")
        return self.he_accuracy

    def analyze_mceliece_matrix(self) -> Tuple[float, int]:
        """
        分析McEliece的稀疏奇偶校验矩阵特性
        返回稀疏度和每行平均非零元素数
        """
        # 获取奇偶校验矩阵（可能需要根据pymceliece库实际API调整）
        parity_matrix = self.mc._parity_check_matrix

        # 计算矩阵稀疏度
        total_elements = parity_matrix.size
        non_zero_elements = np.count_nonzero(parity_matrix)
        sparsity = (1 - non_zero_elements / total_elements) * 100

        # 计算每行平均非零元素数
        row_non_zero = np.count_nonzero(parity_matrix, axis=1)
        avg_non_zero_per_row = np.mean(row_non_zero)

        print(f"McEliece校验矩阵分析: 稀疏度 {sparsity:.2f}%, 每行平均非零元素 {avg_non_zero_per_row:.2f}")
        return sparsity, avg_non_zero_per_row

    def evaluate_error_correction(self) -> float:
        """
        评估McEliece的纠错效果
        需要在使用噪声的process()调用后执行
        """
        if self.total_errors == 0:
            print("请先使用带噪声的process()调用再评估纠错效果")
            return 0.0

        # 获取纠错后的实际错误数
        processed_bytes = pickle.dumps(self.processed_data, protocol=self.serialize_protocol)
        remaining_errors = sum(1 for o, d in zip(self.original_data, processed_bytes) if o != d)

        # 计算纠错成功率
        corrected_errors = self.total_errors - remaining_errors
        self.error_correction_rate = (corrected_errors / self.total_errors) * 100

        print(
            f"纠错效果评估: 总错误 {self.total_errors}, 纠正 {corrected_errors}, 成功率 {self.error_correction_rate:.2f}%")
        return self.error_correction_rate

    def get_parameters(self) -> dict:
        """返回当前配置的所有参数"""
        return {
            'he_poly_modulus_degree': self.he_poly_modulus_degree,
            'he_plain_modulus': self.he_plain_modulus,
            'he_generate_galois': self.he_generate_galois,
            'mc_n': self.mc_n,
            'mc_k': self.mc_k,
            'mc_t': self.mc_t,
            'redundancy_ratio': self.redundancy_ratio,
            'compress_level': self.compress_level,
            'serialize_protocol': self.serialize_protocol,
            'noise_probability': self.noise_probability,
            'max_noise_intensity': self.max_noise_intensity,
            'he_accuracy': self.he_accuracy,
            'error_correction_rate': self.error_correction_rate
        }

