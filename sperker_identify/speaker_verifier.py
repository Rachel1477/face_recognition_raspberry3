"""
高精度声纹比对脚本
使用 wespeaker + silero-vad 实现声纹识别

Author: sperker_com
"""

import os
import logging
import numpy as np
import torch
import torchaudio
from scipy.signal import resample_poly
from typing import Optional, Tuple, Union
import warnings

# 忽略警告
warnings.filterwarnings('ignore')

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


class SpeakerVerifier:
    """
    声纹验证器类
    使用 WeSpeaker 的预训练模型进行声纹嵌入提取，
    使用 Silero VAD 进行语音活动检测。
    """
    
    # 默认参数
    DEFAULT_SAMPLE_RATE = 16000
    MIN_VOICE_DURATION = 0.5  # 最小有效语音时长（秒）
    DEFAULT_THRESHOLD = 0.30  # 默认相似度阈值
    
    # 预训练模型映射
    MODEL_CONFIGS = {
        'chinese_v2': {
            'model_name': 'chinese_v2',
            'model_type': 'ecapa_tdnn',
            'description': '中文预训练模型 (ECAPA-TDNN)'
        },
        'voxceleb': {
            'model_name': 'voxceleb_resnet34',
            'model_type': 'resnet',
            'description': 'VoxCeleb英文模型 (ResNet34)'
        }
    }
    
    def __init__(
        self,
        model_name: str = 'chinese_v2',
        threshold: float = DEFAULT_THRESHOLD,
        device: Optional[str] = None,
        num_threads: int = 4,
        log_level: int = logging.INFO
    ):
        """
        初始化声纹验证器
        
        Args:
            model_name: 预训练模型名称 ('chinese_v2' 或 'voxceleb')
            threshold: 相似度判断阈值 (推荐范围 0.25-0.35)
            device: 计算设备 ('cpu', 'cuda', 'mps', None自动选择)
            num_threads: CPU推理线程数
            log_level: 日志级别 (logging.DEBUG/INFO/WARNING/ERROR)
        """
        # 设置日志级别
        logger.setLevel(log_level)
        
        self.threshold = threshold
        self.model_name = model_name
        
        # 设置设备
        if device is None:
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'
        else:
            self.device = device
            
        # 设置CPU线程数
        if self.device == 'cpu' and num_threads > 0:
            torch.set_num_threads(num_threads)
            
        logger.info(f"[初始化] 使用设备: {self.device}")
        
        # 加载模型
        self._load_wespeaker_model()
        # 加载VAD模型
        self._load_silero_vad()
        
        logger.info(f"[初始化] 声纹验证器初始化完成 (阈值: {self.threshold})")
    
    def _load_wespeaker_model(self):
        """加载WeSpeaker预训练模型"""
        try:
            import wespeaker
        except ImportError:
            raise ImportError(
                "未安装 wespeaker 库，请运行: pip install wespeaker"
            )
        
        model_config = self.MODEL_CONFIGS.get(self.model_name)
        if model_config is None:
            raise ValueError(
                f"不支持的模型: {self.model_name}，"
                f"可选: {list(self.MODEL_CONFIGS.keys())}"
            )
        
        logger.info(f"[加载模型] 正在加载 {model_config['description']}...")
        
        # 加载模型
        self.wespeaker_model = wespeaker.load_model(model_config['model_name'])
        logger.info(f"[加载模型] WeSpeaker 模型加载成功")
    
    def _load_silero_vad(self):
        """加载Silero VAD模型"""
        logger.info(f"[加载模型] 正在加载 Silero VAD...")
        
        # 从silero_vad包导入，如果未安装会报错
        try:
            from silero_vad import load_silero_vad, get_speech_timestamps
        except ImportError:
            raise ImportError(
                "未安装 silero_vad 库，请运行: pip install silero-vad"
            )
        
        self.vad_model = load_silero_vad(
            model_name='silero_vad',
            device=self.device
        )
        self.get_speech_timestamps = get_speech_timestamps
        logger.info(f"[加载模型] Silero VAD 加载成功")
    
    def _load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """
        加载并预处理音频文件
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            Tuple[音频数据(numpy), 采样率]
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"音频文件不存在: {audio_path}")
        
        logger.info(f"正在加载音频: {audio_path}")
        
        # 使用torchaudio加载
        waveform, sample_rate = torchaudio.load(audio_path)
        original_shape = waveform.shape
        original_duration = waveform.shape[1] / sample_rate
        
        logger.info(f"  - 原始采样率: {sample_rate}Hz")
        logger.info(f"  - 原始声道数: {original_shape[0]}")
        logger.info(f"  - 原始时长: {original_duration:.2f}s")
        
        # 转换为单声道
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            logger.info(f"  - 已转换为单声道")
        
        # 重采样到16000Hz
        if sample_rate != self.DEFAULT_SAMPLE_RATE:
            # 计算新的长度
            new_length = int(waveform.shape[1] * self.DEFAULT_SAMPLE_RATE / sample_rate)
            waveform = torch.nn.functional.interpolate(
                waveform.unsqueeze(0),
                size=new_length,
                mode='linear',
                align_corners=False
            ).squeeze(0)
            sample_rate = self.DEFAULT_SAMPLE_RATE
            logger.info(f"  - 已重采样至 {sample_rate}Hz")
        
        # 确保是float32类型
        waveform = waveform.to(torch.float32)
        
        # 归一化到[-1, 1]范围
        max_val = torch.abs(waveform).max()
        if max_val > 0:
            waveform = waveform / max_val
        else:
            logger.warning(f"  - 音频幅值为0，可能是静音文件")
        
        # 计算音频统计信息
        audio_np = waveform.squeeze(0).numpy()
        rms = np.sqrt(np.mean(audio_np ** 2))
        peak = np.max(np.abs(audio_np))
        logger.info(f"  - 峰值幅值: {peak:.4f}, RMS: {rms:.4f}")
        
        return audio_np, sample_rate
    
    def _apply_vad(
        self,
        audio: np.ndarray,
        sample_rate: int
    ) -> np.ndarray:
        """
        应用VAD进行语音活动检测，保留人声部分
        
        Args:
            audio: 音频数据
            sample_rate: 采样率
            
        Returns:
            处理后的音频数据（只保留人声部分）
        """
        logger.info("正在执行VAD语音活动检测...")
        logger.info(f"  - 输入音频时长: {len(audio) / sample_rate:.2f}s")
        logger.info(f"  - 采样率: {sample_rate}Hz")
        logger.info(f"  - 采样点数: {len(audio)}")
        
        # 转换为torch tensor
        audio_tensor = torch.from_numpy(audio).float()
        
        # 获取语音时间戳
        speech_timestamps = self.get_speech_timestamps(
            audio_tensor,
            self.vad_model,
            threshold=0.5,
            min_speech_duration_ms=300,
            min_silence_duration_ms=200,
            sampling_rate=sample_rate
        )
        
        if not speech_timestamps:
            logger.error("VAD检测结果: 未找到任何语音片段")
            logger.error("可能原因:")
            logger.error("  1. 音频为纯静音")
            logger.error("  2. 音频噪声过大")
            logger.error("  3. 人声被背景音乐覆盖")
            logger.error("  4. 音频时长太短")
            raise ValueError(
                f"未检测到人声，可能音频为纯静音或噪声过大。"
                f"原始音频时长: {len(audio) / sample_rate:.2f}s"
            )
        
        # 统计VAD结果
        num_segments = len(speech_timestamps)
        total_speech_samples = sum(ts['end'] - ts['start'] for ts in speech_timestamps)
        total_speech_duration = total_speech_samples / sample_rate
        
        logger.info(f"VAD检测结果:")
        logger.info(f"  - 检测到 {num_segments} 个语音片段")
        logger.info(f"  - 总语音时长: {total_speech_duration:.2f}s")
        logger.info(f"  - 语音占比: {total_speech_duration / (len(audio) / sample_rate) * 100:.1f}%")
        
        # 记录每个片段的信息
        for i, ts in enumerate(speech_timestamps[:5]):  # 最多显示前5个片段
            start_time = ts['start'] / sample_rate
            end_time = ts['end'] / sample_rate
            duration = end_time - start_time
            logger.debug(f"    片段{i+1}: {start_time:.2f}s - {end_time:.2f}s (时长: {duration:.2f}s)")
        if num_segments > 5:
            logger.debug(f"    ... 还有 {num_segments - 5} 个片段")
        
        # 合并所有语音片段
        speech_parts = []
        for ts in speech_timestamps:
            start = ts['start']
            end = ts['end']
            speech_parts.append(audio[start:end])
        
        # 拼接所有语音片段
        voice_audio = np.concatenate(speech_parts)
        
        logger.info(f"  - VAD后音频长度: {len(voice_audio)} 采样点 ({len(voice_audio) / sample_rate:.2f}s)")
        
        return voice_audio
    
    def get_embedding(self, audio_path: str) -> Optional[np.ndarray]:
        """
        提取音频的声纹嵌入向量
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            声纹嵌入向量，如果音频过短则返回None
        """
        logger.info("=" * 40)
        logger.info(f"开始处理音频: {audio_path}")
        
        try:
            # 1. 加载并预处理音频
            audio, sample_rate = self._load_audio(audio_path)
            
            # 2. 应用VAD提取人声
            voice_audio = self._apply_vad(audio, sample_rate)
            
            # 3. 检查有效音频时长
            duration = len(voice_audio) / sample_rate
            logger.info(f"有效语音时长检查: {duration:.2f}s (最小要求: {self.MIN_VOICE_DURATION}s)")
            
            if duration < self.MIN_VOICE_DURATION:
                logger.error(f"语音时长不足!")
                logger.error(f"  - 实际时长: {duration:.2f}s")
                logger.error(f"  - 最小要求: {self.MIN_VOICE_DURATION}s")
                logger.error(f"  - 差距: {self.MIN_VOICE_DURATION - duration:.2f}s")
                raise ValueError(
                    f"有效语音时长过短 ({duration:.2f}s)，"
                    f"最少需要 {self.MIN_VOICE_DURATION}s"
                )
            
            # 4. 转换为torch tensor
            voice_tensor = torch.from_numpy(voice_audio).float()
            
            # 5. 提取embedding
            logger.info("正在提取声纹嵌入向量...")
            embedding = self.wespeaker_model.embeddings([voice_tensor])
            
            logger.info(f"声纹嵌入向量提取成功")
            logger.info(f"  - 向量维度: {embedding[0].shape[0]}")
            logger.info(f"  - 向量范数: {np.linalg.norm(embedding[0]):.4f}")
            logger.info("=" * 40)
            
            return embedding[0]
            
        except FileNotFoundError:
            logger.error(f"文件不存在: {audio_path}")
            raise
        except ValueError:
            raise
        except Exception as e:
            logger.error(f"处理音频时发生未知错误: {type(e).__name__}: {str(e)}")
            raise
    
    def _cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        计算两个向量的余弦相似度
        
        Args:
            emb1: 嵌入向量1
            emb2: 嵌入向量2
            
        Returns:
            余弦相似度分数
        """
        # 归一化后点积即为余弦相似度
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return np.dot(emb1, emb2) / (norm1 * norm2)
    
    def verify(
        self,
        audio_path1: str,
        audio_path2: str
    ) -> Tuple[float, bool]:
        """
        验证两段音频是否为同一人
        
        Args:
            audio_path1: 第一个音频文件路径
            audio_path2: 第二个音频文件路径
            
        Returns:
            Tuple[相似度分数, 是否为同一人]
        """
        logger.info("=" * 50)
        logger.info("开始声纹比对")
        logger.info(f"音频1: {audio_path1}")
        logger.info(f"音频2: {audio_path2}")
        logger.info(f"阈值: {self.threshold}")
        
        # 提取两个音频的embedding
        try:
            emb1 = self.get_embedding(audio_path1)
            emb2 = self.get_embedding(audio_path2)
        except Exception as e:
            logger.error(f"提取声纹失败: {str(e)}")
            raise
        
        # 计算余弦相似度
        similarity = self._cosine_similarity(emb1, emb2)
        
        # 判断结果
        is_same = similarity >= self.threshold
        
        logger.info("-" * 50)
        logger.info(f"比对结果:")
        logger.info(f"  - 相似度: {similarity:.4f}")
        logger.info(f"  - 阈值: {self.threshold}")
        logger.info(f"  - 判断: {'同一人 ✓' if is_same else '不同人 ✗'}")
        logger.info("=" * 50)
        
        return similarity, is_same
    
    def batch_verify(
        self,
        reference_path: str,
        test_paths: list
    ) -> list:
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
        
        # 预先提取参考音频的embedding
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
                similarity = self._cosine_similarity(ref_embedding, test_embedding)
                is_same = similarity >= self.threshold
                results.append({
                    'path': test_path,
                    'similarity': similarity,
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
    logger.info("=" * 50)
    logger.info("声纹比对系统 - 示例程序")
    logger.info("=" * 50)
    
    # 创建验证器实例
    verifier = SpeakerVerifier(
        model_name='chinese_v2',  # 使用中文模型
        threshold=0.30,            # 相似度阈值
        num_threads=4             # CPU线程数
    )
    
    # 测试代码 - 请替换为实际的音频文件路径
    # 示例:
    # audio1 = "path/to/audio1.wav"
    # audio2 = "path/to/audio2.wav"
    # similarity, is_same = verifier.verify(audio1, audio2)
    
    logger.info("提示: 请使用 verify() 方法比对两个音频文件")
    logger.info("示例: verifier.verify('audio1.wav', 'audio2.wav')")
    
    # 打印支持的模型列表
    logger.info("支持的预训练模型:")
    for name, config in SpeakerVerifier.MODEL_CONFIGS.items():
        logger.info(f"  - {name}: {config['description']}")


if __name__ == '__main__':
    main()
