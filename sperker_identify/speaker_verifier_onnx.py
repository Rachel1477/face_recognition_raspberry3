"""
高精度声纹比对脚本 (ONNX Runtime 版本)
使用 silero-vad + ONNX ResNet34 模型实现声纹识别

依赖安装:
    pip install onnxruntime silero-vad librosa numpy soundfile
"""

import os
import sys
import logging
import hashlib
import urllib.request
from typing import Optional, Tuple, List

import numpy as np

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


class SpeakerVerifier:
    """
    声纹验证器类 (ONNX Runtime 版本)
    使用 Silero VAD 进行语音活动检测，
    使用 ONNX ResNet34 模型提取声纹嵌入。
    """
    
    # 常量配置
    DEFAULT_SAMPLE_RATE = 16000
    MIN_VOICE_DURATION = 0.5  # 最小有效语音时长（秒）
    DEFAULT_THRESHOLD = 0.30   # 默认相似度阈值
    EMBEDDING_DIM = 256        # 嵌入向量维度
    
    # 模型配置
    MODEL_CONFIG = {
        'model_name': 'wespeaker_zh_cnceleb_resnet34.onnx',
        'model_url': 'https://github.com/wespeakervietnam/wespeaker/raw/master/wespeaker/models/speaker_resnet34_256_common.onnx',
        'embedding_dim': 256,
        'model_type': 'ResNet34'
    }
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        threshold: float = DEFAULT_THRESHOLD,
        num_threads: int = 4,
        log_level: int = logging.INFO
    ):
        """
        初始化声纹验证器
        
        Args:
            model_path: ONNX 模型文件路径，None 则使用默认路径
            threshold: 相似度判断阈值 (推荐范围 0.25-0.35)
            num_threads: CPU 推理线程数
            log_level: 日志级别
        """
        logger.setLevel(log_level)
        
        self.threshold = threshold
        
        # 确定模型路径
        if model_path is None:
            model_dir = os.path.join(os.path.expanduser('~'), '.wespeaker', 'models')
            os.makedirs(model_dir, exist_ok=True)
            self.model_path = os.path.join(model_dir, self.MODEL_CONFIG['model_name'])
        else:
            self.model_path = model_path
        
        # 下载模型（如需要）
        self._ensure_model()
        
        # 加载 ONNX Runtime
        self._load_onnx_model(num_threads)
        
        # 加载 Silero VAD
        self._load_silero_vad()
        
        logger.info(f"[初始化] 声纹验证器初始化完成 (阈值: {self.threshold})")
    
    def _ensure_model(self):
        """确保模型文件存在，如不存在则下载"""
        print(self.model_path)
        if os.path.exists(self.model_path):
            logger.info(f"[模型] 模型已存在: {self.model_path}")
            return
        
        logger.warning(f"[模型] 模型文件不存在，正在下载...")
        logger.warning(f"[模型] 如需手动下载，请访问: {self.MODEL_CONFIG['model_url']}")
        
        model_url = self.MODEL_CONFIG['model_url']
        temp_path = self.model_path + '.tmp'
        
        try:
            # 下载模型
            urllib.request.urlretrieve(model_url, temp_path)
            
            # 移动到目标位置
            if os.path.exists(self.model_path):
                os.remove(self.model_path)
            os.rename(temp_path, self.model_path)
            
            logger.info(f"[模型] 模型下载完成: {self.model_path}")
            
        except Exception as e:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise RuntimeError(
                f"模型下载失败: {str(e)}\n"
                f"请手动下载模型并放置到: {self.model_path}\n"
                f"下载链接: {model_url}"
            )
    
    def _load_onnx_model(self, num_threads: int):
        """加载 ONNX 模型"""
        logger.info(f"[加载] 正在加载 ONNX 模型: {self.model_path}")
        
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError(
                "未安装 onnxruntime，请运行: pip install onnxruntime"
            )
        
        # 配置会话选项
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = num_threads
        sess_options.inter_op_num_threads = num_threads
        
        # 创建推理会话
        self.ort_session = ort.InferenceSession(
            self.model_path,
            sess_options=sess_options,
            providers=['CPUExecutionProvider']
        )
        
        # 获取模型输入输出名称
        self.input_name = self.ort_session.get_inputs()[0].name
        self.output_name = self.ort_session.get_outputs()[0].name
        
        logger.info(f"[加载] ONNX 模型加载成功")
        logger.info(f"[加载] 输入节点: {self.input_name}")
        logger.info(f"[加载] 输出节点: {self.output_name}")
    
    def _load_silero_vad(self):
        """加载 Silero VAD 模型"""
        logger.info(f"[加载] 正在加载 Silero VAD...")
        
        try:
            from silero_vad import load_silero_vad, get_speech_timestamps
        except ImportError:
            raise ImportError(
                "未安装 silero_vad，请运行: pip install silero-vad"
            )
        
        self.vad_model = load_silero_vad()
        self.get_speech_timestamps = get_speech_timestamps
        
        logger.info(f"[加载] Silero VAD 加载成功")
    
    def _load_audio(self, audio_path: str) -> np.ndarray:
        """
        加载并预处理音频文件
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            音频数据 (numpy array, float32)
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"音频文件不存在: {audio_path}")
        
        logger.info(f"正在加载音频: {audio_path}")
        
        import librosa
        
        audio, sr = librosa.load(audio_path, sr=self.DEFAULT_SAMPLE_RATE, mono=True)
        
        logger.info(f"  - 采样率: {sr}Hz")
        logger.info(f"  - 时长: {len(audio) / sr:.2f}s")
        
        # 确保是 float32
        audio = audio.astype(np.float32)
        
        # 归一化到 [-1, 1]
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val
        
        # 计算音频统计
        rms = np.sqrt(np.mean(audio ** 2))
        peak = np.max(np.abs(audio))
        logger.info(f"  - 峰值: {peak:.4f}, RMS: {rms:.4f}")
        
        return audio
    
    def _apply_vad(self, audio: np.ndarray) -> Optional[np.ndarray]:
        """
        应用 VAD 进行语音活动检测
        
        Args:
            audio: 音频数据
            
        Returns:
            处理后的音频数据（只保留人声部分），如果未检测到语音则返回 None
        """
        logger.info("正在执行 VAD 语音活动检测...")
        logger.info(f"  - 输入音频时长: {len(audio) / self.DEFAULT_SAMPLE_RATE:.2f}s")
        
        # 将 numpy 数组转换为 torch tensor
        import torch
        audio_tensor = torch.from_numpy(audio).float()
        
        # 获取语音时间戳
        speech_timestamps = self.get_speech_timestamps(
            audio_tensor,
            self.vad_model
        )
        
        if not speech_timestamps:
            logger.warning("VAD 检测结果: 未找到任何语音片段")
            logger.warning("可能原因: 音频为纯静音、噪声过大或人声被背景音乐覆盖")
            return None
        
        # 统计 VAD 结果
        num_segments = len(speech_timestamps)
        total_samples = sum(ts['end'] - ts['start'] for ts in speech_timestamps)
        total_duration = total_samples / self.DEFAULT_SAMPLE_RATE
        
        logger.info(f"VAD 检测结果:")
        logger.info(f"  - 检测到 {num_segments} 个语音片段")
        logger.info(f"  - 总语音时长: {total_duration:.2f}s")
        logger.info(f"  - 语音占比: {total_duration / (len(audio) / self.DEFAULT_SAMPLE_RATE) * 100:.1f}%")
        
        # 合并语音片段
        speech_parts = []
        for ts in speech_timestamps:
            speech_parts.append(audio[ts['start']:ts['end']])
        
        voice_audio = np.concatenate(speech_parts)
        logger.info(f"  - VAD 后音频长度: {len(voice_audio)} 采样点 ({len(voice_audio) / self.DEFAULT_SAMPLE_RATE:.2f}s)")
        
        return voice_audio
    
    def _numpy_to_torch(self, audio: np.ndarray):
        """将 numpy 数组转换为 torch tensor"""
        try:
            import torch
            return torch.from_numpy(audio).float()
        except ImportError:
            raise ImportError("需要安装 torch: pip install torch")
    
    def _preprocess_for_model(self, audio: np.ndarray) -> np.ndarray:
        """
        预处理音频用于模型推理
        
        Args:
            audio: 音频数据
            
        Returns:
            预处理后的音频 (1D array)
        """
        # 确保是 float32 类型
        audio = audio.astype(np.float32)
        
        # 如果音频太短，用静音填充
        min_samples = int(0.5 * self.DEFAULT_SAMPLE_RATE)  # 最少 0.5 秒
        if len(audio) < min_samples:
            logger.warning(f"音频过短 ({len(audio)} samples)，进行填充")
            audio = np.pad(audio, (0, min_samples - len(audio)), mode='constant')
        
        return audio
    
    def _compute_fbank(self, audio: np.ndarray) -> np.ndarray:
        """
        计算 Fbank 特征 (80 维) - Kaldi 风格
        
        Args:
            audio: 音频数据 (16kHz)
            
        Returns:
            Fbank 特征 [T, 80]
        """
        logger.info("正在计算 Fbank 特征 (Kaldi 风格)...")
        
        import librosa
        
        # 参数配置 (Kaldi 风格)
        n_mels = 80
        n_fft = 512
        hop_length = 160
        pre_emphasis = 0.97
        
        # 1. Pre-emphasis (预加重)
        # y[t] = x[t] - pre_emphasis * x[t-1]
        audio_emph = np.append(audio[0], audio[1:] - pre_emphasis * audio[:-1])
        
        # 2. 计算 Mel 频谱 (Kaldi 风格参数)
        mel_spec = librosa.feature.melspectrogram(
            y=audio_emph,
            sr=self.DEFAULT_SAMPLE_RATE,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            window='hamming',      # Hamming 窗
            center=False,          # 不居中，Kaldi 风格
            power=2.0,             # 功率谱
            fmin=20,               # 人声下限 20Hz
            fmax=7600              # 人声上限 7600Hz
        )
        
        # 3. 取对数
        log_mel = np.log(mel_spec + 1e-6)
        
        # 4. 转置为 [T, 80]
        feat = log_mel.T
        
        logger.info(f"  - 特征维度: {feat.shape}")
        logger.info(f"  - Pre-emphasis: {pre_emphasis}")
        logger.info(f"  - Mel 范围: 20Hz - 7600Hz")
        
        return feat.astype(np.float32)
    
    def _apply_cmvn(self, feat: np.ndarray) -> np.ndarray:
        """
        应用 CMVN (均值方差归一化)
        
        Args:
            feat: Fbank 特征 [T, 80]
            
        Returns:
            归一化后的特征 [T, 80]
        """
        # 计算均值和方差
        mean = np.mean(feat, axis=0, keepdims=True)
        var = np.var(feat, axis=0, keepdims=True)
        
        # 避免除以零
        eps = 1e-8
        std = np.sqrt(var + eps)
        
        # 归一化
        feat_norm = (feat - mean) / std
        
        return feat_norm.astype(np.float32)
    
    def _extract_embedding(self, audio: np.ndarray) -> np.ndarray:
        """
        使用 ONNX 模型提取声纹嵌入
        
        Args:
            audio: 预处理后的音频数据
            
        Returns:
            256 维声纹嵌入向量
        """
        logger.info("正在提取声纹嵌入...")
        
        # 1. 计算 Fbank 特征 [T, 80]
        fbank_feat = self._compute_fbank(audio)
        
        # 2. 应用 CMVN
        feat_norm = self._apply_cmvn(fbank_feat)
        
        # 3. 扩展维度为 [1, T, 80]
        feat_input = np.expand_dims(feat_norm, axis=0)
        
        logger.info(f"  - 输入形状: {feat_input.shape}")
        
        # ONNX 推理
        embedding = self.ort_session.run(
            [self.output_name],
            {self.input_name: feat_input}
        )[0]
        
        # 输出通常是 [1, 256]，取第一行并归一化
        embedding = embedding[0]
        embedding = embedding / np.linalg.norm(embedding)
        
        logger.info(f"  - 向量维度: {len(embedding)}")
        logger.info(f"  - 向量范数: {np.linalg.norm(embedding):.4f}")
        
        return embedding
    
    def get_embedding(self, audio_path: str) -> Optional[np.ndarray]:
        """
        提取音频的声纹嵌入向量
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            256 维声纹嵌入向量，如果音频无效则返回 None
        """
        logger.info("=" * 40)
        logger.info(f"开始处理音频: {audio_path}")
        
        try:
            # 1. 加载音频 (使用 librosa 直接加载为 16000Hz)
            audio = self._load_audio(audio_path)
            
            # 2. VAD 切除静音
            voice_audio = self._apply_vad(audio)
            
            # 如果 VAD 未检测到语音，返回 None
            if voice_audio is None:
                logger.warning(f"VAD 未检测到语音，返回 None")
                logger.info("=" * 40)
                return None
            
            # 3. 检查有效音频时长
            duration = len(voice_audio) / self.DEFAULT_SAMPLE_RATE
            logger.info(f"有效语音时长检查: {duration:.2f}s (最小要求: {self.MIN_VOICE_DURATION}s)")
            
            if duration < self.MIN_VOICE_DURATION:
                logger.warning(f"语音时长不足: {duration:.2f}s < {self.MIN_VOICE_DURATION}s")
                logger.info("=" * 40)
                return None
            
            # 4. 预处理
            processed_audio = self._preprocess_for_model(voice_audio)
            
            # 5. 提取嵌入
            embedding = self._extract_embedding(processed_audio)
            
            logger.info(f"声纹嵌入提取成功")
            logger.info("=" * 40)
            
            return embedding
            
        except FileNotFoundError:
            logger.error(f"文件不存在: {audio_path}")
            return None
        except Exception as e:
            logger.error(f"处理音频时发生错误: {type(e).__name__}: {str(e)}")
            return None
    
    @staticmethod
    def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        计算两个向量的余弦相似度
        
        Args:
            emb1: 嵌入向量 1
            emb2: 嵌入向量 2
            
        Returns:
            余弦相似度分数
        """
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(np.dot(emb1, emb2) / (norm1 * norm2))
    
    def verify(self, audio_path1: str, audio_path2: str) -> Tuple[float, bool]:
        """
        验证两段音频是否为同一人
        
        Args:
            audio_path1: 第一个音频文件路径
            audio_path2: 第二个音频文件路径
            
        Returns:
            Tuple[相似度分数, 是否为同一人]，如果任一音频无效则返回 (0.0, False)
        """
        logger.info("=" * 50)
        logger.info("开始声纹比对")
        logger.info(f"音频1: {audio_path1}")
        logger.info(f"音频2: {audio_path2}")
        logger.info(f"阈值: {self.threshold}")
        
        # 提取两个音频的嵌入
        emb1 = self.get_embedding(audio_path1)
        emb2 = self.get_embedding(audio_path2)
        
        # 检查嵌入是否有效
        if emb1 is None:
            logger.error(f"无法提取音频1的声纹: {audio_path1}")
            logger.info("=" * 50)
            return 0.0, False
        
        if emb2 is None:
            logger.error(f"无法提取音频2的声纹: {audio_path2}")
            logger.info("=" * 50)
            return 0.0, False
        
        # 计算余弦相似度
        similarity = self.cosine_similarity(emb1, emb2)
        
        # 判断结果
        is_same = similarity >= self.threshold
        
        logger.info("-" * 50)
        logger.info(f"比对结果:")
        logger.info(f"  - 相似度: {similarity:.4f}")
        logger.info(f"  - 阈值: {self.threshold}")
        logger.info(f"  - 判断: {'同一人 ✓' if is_same else '不同人 ✗'}")
        logger.info("=" * 50)
        
        return similarity, is_same
    
    def batch_verify(self, reference_path: str, test_paths: List[str]) -> List[dict]:
        """
        批量验证 - 将多个音频与参考音频比对
        
        Args:
            reference_path: 参考音频路径
            test_paths: 待测试音频路径列表
            
        Returns:
            每个测试音频的比对结果列表
        """
        logger.info("=" * 50)
        logger.info("开始批量声纹比对")
        logger.info(f"参考音频: {reference_path}")
        logger.info(f"待测音频数量: {len(test_paths)}")
        
        # 提取参考音频的嵌入
        try:
            ref_embedding = self.get_embedding(reference_path)
        except Exception as e:
            logger.error(f"提取参考音频声纹失败: {str(e)}")
            raise
        
        results = []
        for i, test_path in enumerate(test_paths, 1):
            logger.info(f"\n批量进度: {i}/{len(test_paths)}")
            try:
                test_embedding = self.get_embedding(test_path)
                similarity = self.cosine_similarity(ref_embedding, test_embedding)
                is_same = similarity >= self.threshold
                results.append({
                    'path': test_path,
                    'similarity': float(similarity),
                    'is_same': is_same
                })
                logger.info(f"  结果: 相似度={similarity:.4f}, {'同一人' if is_same else '不同人'}")
            except Exception as e:
                logger.warning(f"处理失败: {test_path}, 错误: {str(e)}")
                results.append({
                    'path': test_path,
                    'similarity': 0.0,
                    'is_same': False,
                    'error': str(e)
                })
        
        # 统计结果
        success_count = sum(1 for r in results if 'error' not in r)
        same_count = sum(1 for r in results if r.get('is_same', False))
        logger.info(f"\n批量比对完成: 成功 {success_count}/{len(test_paths)}, 同一人 {same_count}个")
        
        return results


def main():
    """示例用法"""
    import argparse
    import logging as log_module
    
    parser = argparse.ArgumentParser(description='声纹比对工具 (ONNX Runtime 版本)')
    parser.add_argument('audio1', help='第一个音频文件路径')
    parser.add_argument('audio2', help='第二个音频文件路径')
    parser.add_argument('--model', '-m', help='ONNX 模型文件路径')
    parser.add_argument('--threshold', '-t', type=float, default=0.30, help='相似度阈值 (默认: 0.30)')
    parser.add_argument('--threads', '-n', type=int, default=4, help='CPU 线程数 (默认: 4)')
    parser.add_argument('--verbose', '-v', action='store_true', help='显示详细日志')
    
    args = parser.parse_args()
    
    # 设置日志级别
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.getLogger().setLevel(log_level)
    
    # 创建验证器
    verifier = SpeakerVerifier(
        model_path=args.model,
        threshold=args.threshold,
        num_threads=args.threads,
        log_level=log_level
    )
    
    # 执行比对
    similarity, is_same = verifier.verify(args.audio1, args.audio2)
    
    print(f"\n最终结果:")
    print(f"  相似度: {similarity:.4f}")
    print(f"  判断: {'同一人' if is_same else '不同人'}")


if __name__ == '__main__':
    main()
