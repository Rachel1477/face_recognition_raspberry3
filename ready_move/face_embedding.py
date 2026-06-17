"""
人脸检测与识别模块
使用 OpenCV 官方接口：
- cv2.FaceDetectorYN: 人脸检测
- cv2.FaceRecognizerSF: 人脸识别
"""
import numpy as np
import cv2
from typing import Optional, Tuple, List
import logging

logger = logging.getLogger(__name__)


class FaceRecognizer:
    """OpenCV 官方人脸识别器（使用 FaceDetectorYN + FaceRecognizerSF）"""

    def __init__(self, detector_model: str, recognizer_model: str,
                 confidence_threshold: float = 0.9, nms_threshold: float = 0.3):
        """
        初始化人脸识别器

        Args:
            detector_model: YuNet 检测器模型路径
            recognizer_model: SFace 识别器模型路径
            confidence_threshold: 检测置信度阈值
            nms_threshold: NMS 阈值
        """
        self.detector_model = detector_model
        self.recognizer_model = recognizer_model
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold

        # 检查 OpenCV 版本（需要 4.7.0+）
        major, minor, _ = cv2.__version__.split('.')
        if int(major) < 4 or (int(major) == 4 and int(minor) < 7):
            raise RuntimeError(
                f"OpenCV 版本 {cv2.__version__} 不支持 Face 模块，需要 4.7.0+"
            )

        # 创建检测器
        logger.info(f"加载 YuNet 检测器: {detector_model}")
        self.detector = cv2.FaceDetectorYN.create(
            detector_model,
            "",
            (320, 320),
            confidence_threshold,
            nms_threshold
        )

        # 创建识别器
        logger.info(f"加载 SFace 识别器: {recognizer_model}")
        self.recognizer = cv2.FaceRecognizerSF.create(
            recognizer_model,
            ""
        )

        logger.info("人脸识别器初始化完成")

    def detect(self, image: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        检测人脸

        Args:
            image: BGR 格式图像

        Returns:
            (faces, landmarks) 或 None
            - faces: 检测到的人脸框，shape=(N, 15)，包含坐标和置信度
            - landmarks: 人脸关键点，shape=(N, 10)，5个关键点
        """
        h, w = image.shape[:2]

        # 设置输入图像尺寸
        self.detector.setInputSize((w, h))

        # 检测人脸
        success, faces = self.detector.detect(image)

        if not success or faces is None or len(faces) == 0:
            return None

        # 解析结果
        results = []
        for face in faces:
            # face 格式: [x, y, w, h, confidence, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5]
            # 前4个: 边界框 (x, y, w, h)
            # 第5个: 置信度
            # 后10个: 5个关键点坐标

            bbox = face[:4].astype(int)
            confidence = face[4]
            landmarks = face[5:].reshape(5, 2)

            results.append((bbox, landmarks, confidence))

        return results

    def recognize(self, image: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        检测并识别单个人脸（返回置信度最高的人脸）

        Args:
            image: BGR 格式图像

        Returns:
            (aligned_face, feature) 或 None
            - aligned_face: 对齐后的人脸图像 (3, 112, 112)
            - feature: 128维特征向量
        """
        detection_result = self.detect(image)

        if detection_result is None:
            return None

        # 取置信度最高的人脸
        best_face = max(detection_result, key=lambda x: x[2])
        bbox, landmarks, confidence = best_face

        # 人脸对齐并裁剪
        aligned_face = self.recognizer.alignCrop(image, landmarks)

        # 提取特征
        feature = self.recognizer.feature(aligned_face)

        return aligned_face, feature.flatten()

    def recognize_all(self, image: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        """
        检测并识别所有人脸

        Args:
            image: BGR 格式图像

        Returns:
            [(aligned_face, feature, confidence), ...]
        """
        detection_result = self.detect(image)

        if detection_result is None:
            return []

        results = []
        for bbox, landmarks, confidence in detection_result:
            try:
                aligned_face = self.recognizer.alignCrop(image, landmarks)
                feature = self.recognizer.feature(aligned_face)
                results.append((aligned_face, feature.flatten(), confidence))
            except Exception as e:
                logger.warning(f"人脸特征提取失败: {e}")
                continue

        return results

    @staticmethod
    def cosine_similarity(feature1: np.ndarray, feature2: np.ndarray) -> float:
        """
        计算余弦相似度

        Args:
            feature1: 特征向量 1
            feature2: 特征向量 2

        Returns:
            余弦相似度 [-1, 1]
        """
        if feature1 is None or feature2 is None:
            return 0.0

        f1 = feature1.flatten().astype(np.float32)
        f2 = feature2.flatten().astype(np.float32)

        norm1 = np.linalg.norm(f1)
        norm2 = np.linalg.norm(f2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return np.dot(f1, f2) / (norm1 * norm2)


def create_face_recognizer(detector_model: str, recognizer_model: str,
                           confidence_threshold: float = 0.9,
                           nms_threshold: float = 0.3) -> FaceRecognizer:
    """
    创建人脸识别器

    Args:
        detector_model: 检测器模型路径
        recognizer_model: 识别器模型路径
        confidence_threshold: 置信度阈值
        nms_threshold: NMS 阈值

    Returns:
        FaceRecognizer 实例
    """
    return FaceRecognizer(
        detector_model,
        recognizer_model,
        confidence_threshold,
        nms_threshold
    )
