"""
人脸识别工具模块 - PC 后端版本
使用 insightface 库进行高精度人脸识别
"""
import os
import cv2
import numpy as np
from typing import List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
import threading
import logging
import json

logger = logging.getLogger(__name__)

# 尝试导入 insightface
try:
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    logger.warning("insightface 未安装，将使用 OpenCV 备选方案")


class FaceProcessor:
    """
    人脸识别处理器（PC 后端版本）
    使用 insightface 的 buffalo_l 模型，支持高进度人脸识别
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        """单例模式，确保只有一个实例"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """初始化人脸识别处理器"""
        if hasattr(self, '_initialized'):
            return
        
        self._initialized = True
        self.app = None
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        if INSIGHTFACE_AVAILABLE:
            self._init_insightface()
        else:
            self._init_opencv_fallback()
    
    def _init_insightface(self):
        """初始化 insightface 模型"""
        try:
            # 初始化 FaceAnalysis，使用 buffalo_l 模型
            self.app = FaceAnalysis(
                name='buffalo_l',  # 使用 buffalo_l 模型，精度最高
                providers=['CPUExecutionProvider']
            )
            self.app.prepare(ctx_id=0, det_size=(640, 640))
            logger.info("insightface buffalo_l 模型加载成功")
        except Exception as e:
            logger.error(f"insightface 初始化失败: {e}")
            self._init_opencv_fallback()
    
    def _init_opencv_fallback(self):
        """初始化 OpenCV 备选方案"""
        try:
            from app.config import MODEL_DIR
            
            detector_path = os.path.join(MODEL_DIR, "face_detection_yunet_2023mar.onnx")
            recognizer_path = os.path.join(MODEL_DIR, "face_recognition_sface_2021dec.onnx")
            
            if os.path.exists(detector_path) and os.path.exists(recognizer_path):
                import os
                detector_path = os.path.abspath(detector_path)
                recognizer_path = os.path.abspath(recognizer_path)
                
                self.detector = cv2.FaceDetectorYN.create(detector_path, "", (640, 640))
                self.recognizer = cv2.FaceRecognizerSF.create(recognizer_path, "")
                logger.info("OpenCV SFace 模型加载成功（备选方案）")
            else:
                logger.error("找不到 ONNX 模型文件")
                self.app = None
                self.detector = None
                self.recognizer = None
        except Exception as e:
            logger.error(f"OpenCV 备选方案初始化失败: {e}")
            self.app = None
            self.detector = None
            self.recognizer = None
    
    def extract_face_features(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        提取人脸特征向量
        
        Args:
            image: BGR 格式的图像
            
        Returns:
            128 维特征向量，或 None（未检测到人脸）
        """
        if self.app is not None:
            return self._extract_with_insightface(image)
        elif self.detector is not None and self.recognizer is not None:
            return self._extract_with_opencv(image)
        else:
            logger.error("人脸识别模型未初始化")
            return None

    def detect_faces(self, image: np.ndarray) -> List[dict]:
        """
        检测图像中所有人脸的位置（用于调试）
        
        Args:
            image: BGR 格式的图像
            
        Returns:
            [{'bbox': [x1, y1, x2, y2], 'confidence': float, 'embedding': array}, ...]
        """
        results = []
        
        if self.app is not None:
            try:
                faces = self.app.get(image)
                if faces:
                    for face in faces:
                        bbox = face.bbox.astype(int).tolist()  # [x1, y1, x2, y2]
                        confidence = float(face.det_score) if hasattr(face, 'det_score') else 0.0
                        embedding = face.embedding
                        if embedding is not None:
                            norm = np.linalg.norm(embedding)
                            if norm > 0:
                                embedding = embedding / norm
                        results.append({
                            'bbox': bbox,
                            'confidence': round(confidence, 4),
                            'embedding': embedding
                        })
            except Exception as e:
                logger.error(f"insightface 人脸检测失败: {e}")
        elif self.detector is not None:
            try:
                h, w = image.shape[:2]
                self.detector.setInputSize((w, h))
                success, faces = self.detector.detect(image)
                if success and faces is not None:
                    for face in faces:
                        bbox = face[:4].astype(int).tolist()
                        confidence = float(face[4])
                        results.append({
                            'bbox': bbox,
                            'confidence': round(confidence, 4),
                            'embedding': None
                        })
            except Exception as e:
                logger.error(f"OpenCV 人脸检测失败: {e}")
        
        return results

    def draw_face_box(self, image: np.ndarray, bbox: list, 
                       label: str = "", color: tuple = (0, 255, 0)) -> np.ndarray:
        """
        在图片上绘制人脸框和标签
        
        Args:
            image: BGR 格式的图像
            bbox: [x1, y1, x2, y2]
            label: 标签文字
            color: BGR 颜色元组，绿色=(0,255,0)，红色=(0,0,255)
            
        Returns:
            绘制后的图片
        """
        img = image.copy()
        x1, y1, x2, y2 = bbox
        
        # 绘制矩形框
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # 绘制标签背景
        if label:
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(img, (x1, y1 - h - 10), (x1 + w, y1), color, -1)
            cv2.putText(img, label, (x1, y1 - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return img
    
    def _extract_with_insightface(self, image: np.ndarray) -> Optional[np.ndarray]:
        """使用 insightface 提取特征"""
        try:
            # 检测人脸
            faces = self.app.get(image)
            
            if faces is None or len(faces) == 0:
                logger.debug("未检测到人脸")
                return None
            
            # 取置信度最高的人脸
            faces = sorted(faces, key=lambda x: x.bbox[4] if len(x.bbox) > 4 else 0, reverse=True)
            face = faces[0]
            
            # 提取特征向量
            embedding = face.embedding
            
            if embedding is not None:
                # 归一化
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm
                return embedding.astype(np.float32)
            
            return None
            
        except Exception as e:
            logger.error(f"insightface 特征提取失败: {e}")
            return None
    
    def _extract_with_opencv(self, image: np.ndarray) -> Optional[np.ndarray]:
        """使用 OpenCV SFace 提取特征（备选方案）"""
        try:
            h, w = image.shape[:2]
            self.detector.setInputSize((w, h))
            
            success, faces = self.detector.detect(image)
            
            if not success or faces is None or len(faces) == 0:
                logger.debug("未检测到人脸（OpenCV）")
                return None
            
            # 取置信度最高的人脸
            faces = sorted(faces, key=lambda x: x[4] if len(x) > 4 else 0, reverse=True)
            face = faces[0]
            
            # 人脸对齐
            aligned_face = self.recognizer.alignCrop(image, face)
            
            # 提取特征
            feature = self.recognizer.feature(aligned_face)
            
            if feature is not None:
                # 归一化
                feature = feature.flatten()
                norm = np.linalg.norm(feature)
                if norm > 0:
                    feature = feature / norm
                return feature.astype(np.float32)
            
            return None
            
        except Exception as e:
            logger.error(f"OpenCV 特征提取失败: {e}")
            return None
    
    def extract_all_faces(self, image: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        提取图像中所有人脸的特征向量
        
        Returns:
            [(embedding, bbox), ...]
        """
        results = []
        
        if self.app is not None:
            try:
                faces = self.app.get(image)
                if faces:
                    for face in faces:
                        embedding = face.embedding
                        if embedding is not None:
                            norm = np.linalg.norm(embedding)
                            if norm > 0:
                                embedding = embedding / norm
                            results.append((
                                embedding.astype(np.float32),
                                face.bbox.astype(int)
                            ))
            except Exception as e:
                logger.error(f"insightface 多人脸提取失败: {e}")
        
        return results
    
    def recognize_face(self, image: np.ndarray, known_embeddings: List[np.ndarray], 
                       threshold: float = 0.5) -> Optional[Tuple[int, float]]:
        """
        识别人脸并与已知人脸库比对
        
        Args:
            image: 输入图像
            known_embeddings: 已知人脸特征向量列表 [(user_id, embedding), ...]
            threshold: 余弦相似度阈值
            
        Returns:
            (user_id, similarity) 或 None
        """
        # 提取当前人脸特征
        query_embedding = self.extract_face_features(image)
        
        if query_embedding is None:
            return None
        
        # 与已知人脸比对
        best_match = None
        best_similarity = -1
        
        for user_id, known_embedding in known_embeddings:
            similarity = self.cosine_similarity(query_embedding, known_embedding)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = user_id
        
        if best_similarity >= threshold:
            return (best_match, best_similarity)
        
        return None
    
    @staticmethod
    def cosine_similarity(feature1: np.ndarray, feature2: np.ndarray) -> float:
        """
        计算余弦相似度
        
        Args:
            feature1: 特征向量1
            feature2: 特征向量2
            
        Returns:
            余弦相似度（-1 到 1 之间）
        """
        if feature1 is None or feature2 is None:
            return 0.0
        
        # 确保是 float32 类型
        f1 = feature1.flatten().astype(np.float32)
        f2 = feature2.flatten().astype(np.float32)
        
        # 余弦相似度
        dot_product = np.dot(f1, f2)
        norm_product = np.linalg.norm(f1) * np.linalg.norm(f2)
        
        if norm_product == 0:
            return 0.0
        
        return float(dot_product / norm_product)


# 全局实例（延迟初始化）
_face_processor = None


def get_face_processor() -> FaceProcessor:
    """获取人脸处理器单例"""
    global _face_processor
    if _face_processor is None:
        _face_processor = FaceProcessor()
    return _face_processor


def extract_face_features(image: np.ndarray) -> Optional[np.ndarray]:
    """便捷函数：提取人脸特征"""
    return get_face_processor().extract_face_features(image)


def cosine_similarity(feature1: np.ndarray, feature2: np.ndarray) -> float:
    """便捷函数：计算余弦相似度"""
    return FaceProcessor.cosine_similarity(feature1, feature2)
