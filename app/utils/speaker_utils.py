import os
import sys
import logging
import urllib.request
from typing import Optional, Tuple, List

import numpy as np

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
    
    DEFAULT_SAMPLE_RATE = 16000
    MIN_VOICE_DURATION = 0.5
    DEFAULT_THRESHOLD = 0.60
    EMBEDDING_DIM = 256
    
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
        logger.setLevel(log_level)
        self.threshold = threshold
        
        if model_path is None:
            model_dir = os.path.join(os.path.expanduser('~'), '.wespeaker', 'models')
            os.makedirs(model_dir, exist_ok=True)
            self.model_path = os.path.join(model_dir, self.MODEL_CONFIG['model_name'])
        else:
            self.model_path = model_path
        
        self._ensure_model()
        self._load_onnx_model(num_threads)
        self._load_silero_vad()
        
        logger.info(f"声纹验证器初始化完成 (阈值: {self.threshold})")
    
    def _ensure_model(self):
        if os.path.exists(self.model_path):
            logger.info(f"模型已存在: {self.model_path}")
            return
        
        logger.warning(f"模型文件不存在，正在下载...")
        model_url = self.MODEL_CONFIG['model_url']
        temp_path = self.model_path + '.tmp'
        
        try:
            urllib.request.urlretrieve(model_url, temp_path)
            if os.path.exists(self.model_path):
                os.remove(self.model_path)
            os.rename(temp_path, self.model_path)
            logger.info(f"模型下载完成: {self.model_path}")
        except Exception as e:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise RuntimeError(
                f"模型下载失败: {str(e)}\n"
                f"请手动下载模型并放置到: {self.model_path}\n"
                f"下载链接: {model_url}"
            )
    
    def _load_onnx_model(self, num_threads: int):
        logger.info(f"正在加载 ONNX 模型: {self.model_path}")
        
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("未安装 onnxruntime，请运行: pip install onnxruntime")
        
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = num_threads
        sess_options.inter_op_num_threads = num_threads
        
        self.ort_session = ort.InferenceSession(
            self.model_path,
            sess_options=sess_options,
            providers=['CPUExecutionProvider']
        )
        
        self.input_name = self.ort_session.get_inputs()[0].name
        self.output_name = self.ort_session.get_outputs()[0].name
        
        logger.info(f"ONNX 模型加载成功")
    
    def _load_silero_vad(self):
        logger.info(f"正在加载 Silero VAD...")
        
        try:
            from silero_vad import load_silero_vad, get_speech_timestamps
        except ImportError:
            raise ImportError("未安装 silero_vad，请运行: pip install silero-vad")
        
        self.vad_model = load_silero_vad()
        self.get_speech_timestamps = get_speech_timestamps
        
        logger.info(f"Silero VAD 加载成功")
    
    def _load_audio(self, audio_path: str) -> np.ndarray:
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"音频文件不存在: {audio_path}")
        
        logger.info(f"正在加载音频: {audio_path}")
        
        import librosa
        audio, sr = librosa.load(audio_path, sr=self.DEFAULT_SAMPLE_RATE, mono=True)
        audio = audio.astype(np.float32)
        
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val
        
        return audio
    
    def _apply_vad(self, audio: np.ndarray) -> Optional[np.ndarray]:
        logger.info("正在执行 VAD 语音活动检测...")
        
        import torch
        audio_tensor = torch.from_numpy(audio).float()
        
        speech_timestamps = self.get_speech_timestamps(audio_tensor, self.vad_model)
        
        if not speech_timestamps:
            logger.warning("VAD 检测结果: 未找到任何语音片段")
            return None
        
        speech_parts = []
        for ts in speech_timestamps:
            speech_parts.append(audio[ts['start']:ts['end']])
        
        voice_audio = np.concatenate(speech_parts)
        return voice_audio
    
    def _preprocess_for_model(self, audio: np.ndarray) -> np.ndarray:
        audio = audio.astype(np.float32)
        
        min_samples = int(0.5 * self.DEFAULT_SAMPLE_RATE)
        if len(audio) < min_samples:
            audio = np.pad(audio, (0, min_samples - len(audio)), mode='constant')
        
        return audio
    
    def _compute_fbank(self, audio: np.ndarray) -> np.ndarray:
        logger.info("正在计算 Fbank 特征...")
        
        import librosa
        
        n_mels = 80
        n_fft = 512
        hop_length = 160
        pre_emphasis = 0.97
        
        audio_emph = np.append(audio[0], audio[1:] - pre_emphasis * audio[:-1])
        
        mel_spec = librosa.feature.melspectrogram(
            y=audio_emph,
            sr=self.DEFAULT_SAMPLE_RATE,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            window='hamming',
            center=False,
            power=2.0,
            fmin=20,
            fmax=7600
        )
        
        log_mel = np.log(mel_spec + 1e-6)
        feat = log_mel.T
        
        return feat.astype(np.float32)
    
    def _apply_cmvn(self, feat: np.ndarray) -> np.ndarray:
        mean = np.mean(feat, axis=0, keepdims=True)
        var = np.var(feat, axis=0, keepdims=True)
        eps = 1e-8
        std = np.sqrt(var + eps)
        feat_norm = (feat - mean) / std
        return feat_norm.astype(np.float32)
    
    def _extract_embedding(self, audio: np.ndarray) -> np.ndarray:
        logger.info("正在提取声纹嵌入...")
        
        fbank_feat = self._compute_fbank(audio)
        feat_norm = self._apply_cmvn(fbank_feat)
        feat_input = np.expand_dims(feat_norm, axis=0)
        
        embedding = self.ort_session.run(
            [self.output_name],
            {self.input_name: feat_input}
        )[0]
        
        embedding = embedding[0]
        embedding = embedding / np.linalg.norm(embedding)
        
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
            audio = self._load_audio(audio_path)
            voice_audio = self._apply_vad(audio)
            
            if voice_audio is None:
                logger.warning(f"VAD 未检测到语音，返回 None")
                logger.info("=" * 40)
                return None
            
            duration = len(voice_audio) / self.DEFAULT_SAMPLE_RATE
            logger.info(f"有效语音时长检查: {duration:.2f}s")
            
            if duration < self.MIN_VOICE_DURATION:
                logger.warning(f"语音时长不足: {duration:.2f}s < {self.MIN_VOICE_DURATION}s")
                logger.info("=" * 40)
                return None
            
            processed_audio = self._preprocess_for_model(voice_audio)
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
    
    def get_embedding_from_bytes(self, audio_bytes: bytes) -> Optional[np.ndarray]:
        """
        从音频字节数据提取声纹嵌入向量
        
        Args:
            audio_bytes: 音频字节数据
            
        Returns:
            256 维声纹嵌入向量，如果音频无效则返回 None
        """
        logger.info("=" * 40)
        logger.info("开始处理音频字节数据")
        
        try:
            import soundfile as sf
            import io
            
            audio_io = io.BytesIO(audio_bytes)
            audio, sr = sf.read(audio_io, dtype='float32')
            
            if sr != self.DEFAULT_SAMPLE_RATE:
                import librosa
                audio = librosa.resample(audio, sr, self.DEFAULT_SAMPLE_RATE)
            
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            
            audio = audio.astype(np.float32)
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                audio = audio / max_val
            
            voice_audio = self._apply_vad(audio)
            
            if voice_audio is None:
                logger.warning(f"VAD 未检测到语音，返回 None")
                logger.info("=" * 40)
                return None
            
            duration = len(voice_audio) / self.DEFAULT_SAMPLE_RATE
            logger.info(f"有效语音时长检查: {duration:.2f}s")
            
            if duration < self.MIN_VOICE_DURATION:
                logger.warning(f"语音时长不足: {duration:.2f}s < {self.MIN_VOICE_DURATION}s")
                logger.info("=" * 40)
                return None
            
            processed_audio = self._preprocess_for_model(voice_audio)
            embedding = self._extract_embedding(processed_audio)
            
            logger.info(f"声纹嵌入提取成功")
            logger.info("=" * 40)
            
            return embedding
            
        except Exception as e:
            logger.error(f"处理音频字节数据时发生错误: {type(e).__name__}: {str(e)}")
            return None
    
    @staticmethod
    def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
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
        
        emb1 = self.get_embedding(audio_path1)
        emb2 = self.get_embedding(audio_path2)
        
        if emb1 is None:
            logger.error(f"无法提取音频1的声纹")
            logger.info("=" * 50)
            return 0.0, False
        
        if emb2 is None:
            logger.error(f"无法提取音频2的声纹")
            logger.info("=" * 50)
            return 0.0, False
        
        similarity = self.cosine_similarity(emb1, emb2)
        is_same = similarity >= self.threshold
        
        logger.info("-" * 50)
        logger.info(f"比对结果:")
        logger.info(f"  - 相似度: {similarity:.4f}")
        logger.info(f"  - 阈值: {self.threshold}")
        logger.info(f"  - 判断: {'同一人 ✓' if is_same else '不同人 ✗'}")
        logger.info("=" * 50)
        
        return similarity, is_same
    
    def verify_with_embedding(self, audio_path: str, reference_embedding: np.ndarray) -> Tuple[float, bool]:
        """
        将音频与已有的声纹嵌入向量比对
        
        Args:
            audio_path: 待验证音频文件路径
            reference_embedding: 参考声纹嵌入向量
            
        Returns:
            Tuple[相似度分数, 是否为同一人]
        """
        logger.info("=" * 50)
        logger.info("开始声纹比对 (使用参考嵌入)")
        logger.info(f"音频: {audio_path}")
        logger.info(f"阈值: {self.threshold}")
        
        emb = self.get_embedding(audio_path)
        
        if emb is None:
            logger.error(f"无法提取音频的声纹")
            logger.info("=" * 50)
            return 0.0, False
        
        similarity = self.cosine_similarity(emb, reference_embedding)
        is_same = similarity >= self.threshold
        
        logger.info("-" * 50)
        logger.info(f"比对结果:")
        logger.info(f"  - 相似度: {similarity:.4f}")
        logger.info(f"  - 阈值: {self.threshold}")
        logger.info(f"  - 判断: {'同一人 ✓' if is_same else '不同人 ✗'}")
        logger.info("=" * 50)
        
        return similarity, is_same
    
    def verify_with_embedding_from_bytes(self, audio_bytes: bytes, reference_embedding: np.ndarray) -> Tuple[float, bool]:
        """
        将音频字节数据与已有的声纹嵌入向量比对
        
        Args:
            audio_bytes: 待验证音频字节数据
            reference_embedding: 参考声纹嵌入向量
            
        Returns:
            Tuple[相似度分数, 是否为同一人]
        """
        logger.info("=" * 50)
        logger.info("开始声纹比对 (使用参考嵌入，字节数据)")
        logger.info(f"阈值: {self.threshold}")
        
        emb = self.get_embedding_from_bytes(audio_bytes)
        
        if emb is None:
            logger.error(f"无法提取音频字节数据的声纹")
            logger.info("=" * 50)
            return 0.0, False
        
        similarity = self.cosine_similarity(emb, reference_embedding)
        is_same = similarity >= self.threshold
        
        logger.info("-" * 50)
        logger.info(f"比对结果:")
        logger.info(f"  - 相似度: {similarity:.4f}")
        logger.info(f"  - 阈值: {self.threshold}")
        logger.info(f"  - 判断: {'同一人 ✓' if is_same else '不同人 ✗'}")
        logger.info("=" * 50)
        
        return similarity, is_same


speaker_verifier = SpeakerVerifier(num_threads=2)