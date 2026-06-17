import cv2
import numpy as np
import json
import os
from typing import Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FaceProcessor:
    """人脸处理器：使用 OpenCV Face 模块进行人脸检测和特征提取"""

    def __init__(self, detector_model: str = None, recognizer_model: str = None):
        """初始化人脸处理器"""
        self.detector = None
        self.recognizer = None
        self._initialize_models(detector_model, recognizer_model)

    def _initialize_models(self, detector_model: str, recognizer_model: str):
        """初始化人脸检测和识别模型"""
        try:
            major, minor, _ = cv2.__version__.split('.')
            if int(major) < 4 or (int(major) == 4 and int(minor) < 7):
                raise RuntimeError(
                    f"OpenCV 版本 {cv2.__version__} 不支持 Face 模块，需要 4.7.0+"
                )

            if detector_model is None:
                detector_model = os.path.abspath(os.path.normpath(os.path.join(
                    os.path.dirname(__file__),
                    "../models/face_detection_yunet_2023mar.onnx"
                )))

            if recognizer_model is None:
                recognizer_model = os.path.abspath(os.path.normpath(os.path.join(
                    os.path.dirname(__file__),
                    "../models/face_recognition_sface_2021dec.onnx"
                )))

            logger.info(f"检测器模型路径: {detector_model}")
            logger.info(f"识别器模型路径: {recognizer_model}")

            if not os.path.exists(detector_model):
                logger.warning(f"检测器模型不存在: {detector_model}")
                logger.info("请下载模型到 app/models/ 目录")

            if not os.path.exists(recognizer_model):
                logger.warning(f"识别器模型不存在: {recognizer_model}")
                logger.info("请下载模型到 app/models/ 目录")

            self.detector = cv2.FaceDetectorYN.create(
                detector_model,
                "",
                (320, 320),
                0.9,
                0.3
            )

            self.recognizer = cv2.FaceRecognizerSF.create(
                recognizer_model,
                ""
            )

            logger.info("SFace 人脸模型初始化成功")

        except Exception as e:
            logger.error(f"人脸模型初始化失败: {e}")
            logger.info("回退到 Haar 级联检测（仅用于测试）")
            self._init_fallback()

    def _init_fallback(self):
        """回退到 Haar 级联检测（用于模型缺失时）"""
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            logger.info("Haar 级联模型初始化成功")
        except Exception as e:
            logger.error(f"Haar 级联模型初始化失败: {e}")

    def detect_face(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        检测图片中的人脸

        Args:
            image: 输入图片（BGR格式）

        Returns:
            人脸位置 (x, y, w, h)，如果没有检测到人脸则返回None
        """
        try:
            if self.detector is not None:
                h, w = image.shape[:2]
                self.detector.setInputSize((w, h))
                success, faces = self.detector.detect(image)

                if not success or faces is None or len(faces) == 0:
                    logger.warning("未检测到人脸")
                    return None

                face = faces[0]
                x, y, w, h = face[:4].astype(int)
                return (x, y, w, h)

            else:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
                )

                if len(faces) == 0:
                    logger.warning("未检测到人脸")
                    return None

                face = max(faces, key=lambda f: f[2] * f[3])
                return tuple(face)

        except Exception as e:
            logger.error(f"人脸检测失败: {e}")
            return None

    def extract_face_features(self, image: np.ndarray) -> Optional[str]:
        """
        提取人脸特征向量（使用 SFace，与树莓派一致）

        Args:
            image: 输入图片（BGR格式）

        Returns:
            128维特征向量的JSON字符串，如果提取失败则返回None
        """
        try:
            if self.detector is not None and self.recognizer is not None:
                h, w = image.shape[:2]
                self.detector.setInputSize((w, h))
                success, faces = self.detector.detect(image)

                if not success or faces is None or len(faces) == 0:
                    logger.warning("无法提取特征：未检测到人脸")
                    return None

                face = faces[0]
                landmarks = face[5:].reshape(5, 2)

                aligned_face = self.recognizer.alignCrop(image, landmarks)

                feature = self.recognizer.feature(aligned_face)

                feature_list = feature.flatten().astype(np.float32).tolist()

                return json.dumps(feature_list)

            else:
                logger.warning("使用回退模式提取特征")
                face_rect = self.detect_face(image)
                if face_rect is None:
                    return None

                x, y, w, h = face_rect
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                face_region = gray[y:y+h, x:x+w]
                face_region = cv2.resize(face_region, (112, 112))
                features = face_region.flatten().tolist()
                features = np.array(features)
                features = (features - features.mean()) / (features.std() + 1e-8)
                return json.dumps(features.tolist())

        except Exception as e:
            logger.error(f"特征提取失败: {e}")
            return None

    def compare_faces(self, feature1: str, feature2: str, threshold: float = 0.36) -> Tuple[bool, float]:
        """
        对比两个人脸特征向量（使用余弦相似度，与树莓派一致）

        Args:
            feature1: 第一个特征向量的JSON字符串
            feature2: 第二个特征向量的JSON字符串
            threshold: 相似度阈值（SFace推荐0.36）

        Returns:
            (是否匹配, 相似度分数)
        """
        try:
            vec1 = np.array(json.loads(feature1))
            vec2 = np.array(json.loads(feature2))

            similarity = self._cosine_similarity(vec1, vec2)

            is_match = similarity >= threshold

            return is_match, float(similarity)

        except Exception as e:
            logger.error(f"人脸对比失败: {e}")
            return False, 0.0

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        计算余弦相似度

        Args:
            vec1: 第一个向量
            vec2: 第二个向量

        Returns:
            相似度分数（-1到1之间）
        """
        if vec1 is None or vec2 is None:
            return 0.0

        f1 = vec1.flatten().astype(np.float32)
        f2 = vec2.flatten().astype(np.float32)

        norm1 = np.linalg.norm(f1)
        norm2 = np.linalg.norm(f2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(np.dot(f1, f2) / (norm1 * norm2))

    def find_best_match(self, target_feature: str, user_features: dict, threshold: float = 0.36) -> Tuple[Optional[int], float]:
        """
        在用户特征库中找到最佳匹配

        Args:
            target_feature: 目标特征向量
            user_features: 用户特征字典 {user_id: feature_json}
            threshold: 相似度阈值

        Returns:
            (最佳匹配的用户ID, 相似度分数)，如果没有匹配则返回(None, 0.0)
        """
        best_user_id = None
        best_similarity = 0.0

        for user_id, feature_json in user_features.items():
            is_match, similarity = self.compare_faces(target_feature, feature_json, threshold)

            if is_match and similarity > best_similarity:
                best_similarity = similarity
                best_user_id = user_id

        return best_user_id, best_similarity


face_processor = FaceProcessor()
