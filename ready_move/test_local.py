"""
本地测试脚本 - 模拟树莓派摄像头输入
用于在没有硬件的情况下测试人脸识别逻辑
"""
import os
import sys
import json
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import glob

import cv2
import numpy as np

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import *
from face_embedding import FaceRecognizer
from gpio_controller import MockGPIOController

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format=LOG_FORMAT
)
logger = logging.getLogger(__name__)


class MockUserCache:
    """模拟用户缓存（用于测试）"""

    def __init__(self):
        self.users: Dict[int, dict] = {}
        self._mock_users = []

    def add_mock_user(self, name: str, embedding: np.ndarray, user_id: int = None):
        """添加模拟用户"""
        if user_id is None:
            user_id = len(self.users) + 1

        self.users[user_id] = {
            'name': name,
            'embedding': embedding
        }
        logger.info(f"添加模拟用户: {name} (ID: {user_id})")

    def find_best_match(self, embedding: np.ndarray, threshold: float = COSINE_THRESHOLD) -> Optional[Tuple[int, str, float]]:
        best_match = None
        best_similarity = -1

        for user_id, user_data in self.users.items():
            similarity = FaceRecognizer.cosine_similarity(
                embedding, user_data['embedding']
            )

            if similarity > best_similarity:
                best_similarity = similarity
                best_match = (user_id, user_data['name'], similarity)

        if best_match and best_similarity >= threshold:
            return best_match

        return None


class ImageCameraSimulator:
    """图片摄像头模拟器 - 从文件夹读取图片模拟摄像头"""

    def __init__(self, image_folder: str):
        """
        初始化图片模拟器

        Args:
            image_folder: 包含测试图片的文件夹路径
        """
        self.image_folder = image_folder
        self.image_files = []
        self.current_index = 0
        self.frame_count = 0

        self._load_images()

    def _load_images(self):
        """加载图片文件"""
        if not os.path.exists(self.image_folder):
            logger.warning(f"图片文件夹不存在: {self.image_folder}")
            return

        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        for ext in extensions:
            pattern = os.path.join(self.image_folder, ext)
            self.image_files.extend(glob.glob(pattern))

        self.image_files.sort()
        logger.info(f"加载 {len(self.image_files)} 张测试图片")

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        模拟摄像头读取

        Returns:
            (success, frame)
        """
        self.frame_count += 1

        if not self.image_files:
            # 如果没有图片，返回空白帧
            return True, np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)

        # 循环播放图片
        if self.current_index >= len(self.image_files):
            self.current_index = 0

        img_path = self.image_files[self.current_index]
        img = cv2.imread(img_path)

        if img is None:
            logger.warning(f"无法读取图片: {img_path}")
            self.current_index += 1
            return True, None

        # 调整图片大小到目标分辨率
        img = cv2.resize(img, (FRAME_WIDTH, FRAME_HEIGHT))

        self.current_index += 1

        return True, img

    def get_next_image(self) -> Optional[np.ndarray]:
        """获取下一张图片"""
        success, frame = self.read()
        return frame if success else None


def check_models():
    """检查模型文件是否存在"""
    missing = []

    if not os.path.exists(FACE_DETECTOR_MODEL):
        missing.append(FACE_DETECTOR_MODEL)
        logger.error(f"缺少检测器模型: {FACE_DETECTOR_MODEL}")
        logger.info("下载命令:")
        logger.info("wget https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx")

    if not os.path.exists(FACE_RECOGNIZER_MODEL):
        missing.append(FACE_RECOGNIZER_MODEL)
        logger.error(f"缺少识别器模型: {FACE_RECOGNIZER_MODEL}")
        logger.info("下载命令:")
        logger.info("wget https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx")

    return len(missing) == 0


def test_face_detection(recognizer: FaceRecognizer, image: np.ndarray) -> bool:
    """测试人脸检测"""
    logger.info("=" * 50)
    logger.info("测试人脸检测...")

    start_time = time.time()

    result = recognizer.recognize(image)

    elapsed = time.time() - start_time

    if result is not None:
        aligned_face, feature = result
        logger.info(f"检测成功!")
        logger.info(f"  - 人脸特征维度: {feature.shape}")
        logger.info(f"  - 特征向量前5个值: {feature[:5]}")
        logger.info(f"  - 处理时间: {elapsed*1000:.1f}ms")

        # 显示对齐后的人脸
        cv2.imshow("Aligned Face", aligned_face)

        return True
    else:
        logger.info("未检测到人脸")
        return False


def test_user_matching(cache: MockUserCache, recognizer: FaceRecognizer, image: np.ndarray):
    """测试用户匹配"""
    logger.info("=" * 50)
    logger.info("测试用户匹配...")

    result = recognizer.recognize(image)

    if result is None:
        logger.info("未检测到人脸，跳过匹配测试")
        return

    aligned_face, feature = result

    if not cache.users:
        logger.info("用户缓存为空，跳过匹配测试")
        logger.info("提示: 使用 --register 参数注册测试用户")
        return

    match_result = cache.find_best_match(feature)

    if match_result:
        user_id, name, similarity = match_result
        logger.info(f"匹配成功!")
        logger.info(f"  - 用户: {name} (ID: {user_id})")
        logger.info(f"  - 相似度: {similarity:.4f}")
        logger.info(f"  - 阈值: {COSINE_THRESHOLD}")
    else:
        logger.info("未匹配到已知用户")
        best_similarity = -1
        best_name = None
        for uid, udata in cache.users.items():
            sim = FaceRecognizer.cosine_similarity(feature, udata['embedding'])
            if sim > best_similarity:
                best_similarity = sim
                best_name = udata['name']
        logger.info(f"  - 最高相似度: {best_similarity:.4f} (用户: {best_name})")
        logger.info(f"  - 阈值: {COSINE_THRESHOLD}")


def register_test_user(cache: MockUserCache, recognizer: FaceRecognizer, image: np.ndarray, name: str = "TestUser"):
    """注册测试用户"""
    logger.info("=" * 50)
    logger.info(f"注册测试用户: {name}")

    result = recognizer.recognize(image)

    if result is None:
        logger.error("未检测到人脸，无法注册")
        return False

    aligned_face, feature = result

    user_id = len(cache.users) + 1
    cache.add_mock_user(name, feature, user_id)

    logger.info(f"注册成功!")
    logger.info(f"  - 用户ID: {user_id}")
    logger.info(f"  - 特征向量: {feature[:5]}... (共{len(feature)}维)")

    return True


def create_test_images():
    """创建测试图片文件夹"""
    test_dir = os.path.join(os.path.dirname(__file__), "test_images")
    os.makedirs(test_dir, exist_ok=True)

    logger.info(f"创建测试图片目录: {test_dir}")
    logger.info("请在该目录放入包含人脸的图片进行测试")
    logger.info("支持格式: jpg, jpeg, png, bmp")

    return test_dir


def main():
    import argparse

    parser = argparse.ArgumentParser(description="人脸识别系统本地测试")
    parser.add_argument("--folder", "-f", type=str, default=None,
                       help="测试图片文件夹路径")
    parser.add_argument("--register", "-r", action="store_true",
                       help="从第一张图片注册测试用户")
    parser.add_argument("--name", "-n", type=str, default="TestUser",
                       help="注册用户名")
    parser.add_argument("--loop", "-l", action="store_true",
                       help="循环播放图片")
    parser.add_argument("--no-gui", action="store_true",
                       help="无GUI模式（用于无显示环境）")
    parser.add_argument("--save-cache", type=str, default=None,
                       help="保存用户缓存到文件")
    parser.add_argument("--load-cache", type=str, default=None,
                       help="从文件加载用户缓存")

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("人脸识别系统本地测试")
    logger.info("=" * 60)

    # 检查 OpenCV 版本
    logger.info(f"OpenCV 版本: {cv2.__version__}")
    major, minor, _ = cv2.__version__.split('.')
    if int(major) < 4 or (int(major) == 4 and int(minor) < 7):
        logger.error("OpenCV 版本需要 4.7.0+ 以支持 Face 模块")
        sys.exit(1)

    # 检查模型文件
    if not check_models():
        logger.error("模型文件缺失，请下载后重试")
        sys.exit(1)

    # 初始化人脸识别器
    try:
        recognizer = FaceRecognizer(
            FACE_DETECTOR_MODEL,
            FACE_RECOGNIZER_MODEL
        )
    except Exception as e:
        logger.error(f"初始化人脸识别器失败: {e}")
        sys.exit(1)

    # 初始化模拟组件
    cache = MockUserCache()
    gpio = MockGPIOController(DOOR_RELAY_PIN, DOOR_UNLOCK_DURATION)

    # 加载或创建用户缓存
    if args.load_cache and os.path.exists(args.load_cache):
        try:
            with open(args.load_cache, 'r') as f:
                data = json.load(f)
                for user_id, user_data in data.items():
                    user_data['embedding'] = np.array(user_data['embedding'])
                    cache.users[int(user_id)] = user_data
            logger.info(f"加载用户缓存: {len(cache.users)} 个用户")
        except Exception as e:
            logger.error(f"加载用户缓存失败: {e}")

    # 创建摄像头模拟器
    if args.folder:
        camera = ImageCameraSimulator(args.folder)
    else:
        test_dir = create_test_images()
        camera = ImageCameraSimulator(test_dir)

    if not camera.image_files:
        logger.warning("没有找到测试图片")

    logger.info("=" * 60)

    # 注册测试用户
    if args.register and camera.image_files:
        first_img = camera.get_next_image()
        if first_img is not None:
            register_test_user(cache, recognizer, first_img, args.name)

    # 主循环
    try:
        while True:
            # 读取帧
            success, frame = camera.read()
            if not success or frame is None:
                logger.error("读取帧失败")
                break

            # 显示原始帧
            if not args.no_gui:
                display = frame.copy()
                cv2.putText(display, f"Frame: {camera.frame_count}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display, f"Users: {len(cache.users)}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                cv2.imshow("Camera Simulation", display)

            # 检测并识别
            result = recognizer.recognize(frame)

            if result is not None:
                aligned_face, feature = result

                logger.info(f"[帧 {camera.frame_count}] 检测到人脸")

                # 用户匹配
                match_result = cache.find_best_match(feature)

                if match_result:
                    user_id, name, similarity = match_result
                    logger.info(f"  -> 识别成功: {name} (相似度: {similarity:.4f})")

                    # 模拟开门
                    gpio.unlock()

                    if not args.no_gui:
                        cv2.putText(display, f"IDENTIFIED: {name}", (10, 90),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    logger.info(f"  -> 未知人员")

                    if not args.no_gui:
                        cv2.putText(display, "UNKNOWN", (10, 90),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                if not args.no_gui:
                    cv2.imshow("Aligned Face", aligned_face)
                    cv2.imshow("Camera Simulation", display)

            # 按键处理
            if not args.no_gui:
                key = cv2.waitKey(100) & 0xFF

                if key == ord('q'):
                    logger.info("退出测试")
                    break
                elif key == ord('r'):
                    # 注册当前帧中的人脸
                    if result is not None:
                        name = input("请输入用户名: ").strip()
                        if name:
                            register_test_user(cache, recognizer, frame, name)
                elif key == ord('s'):
                    # 保存当前人脸特征
                    if result is not None:
                        _, feature = result
                        cache.add_mock_user(f"User_{len(cache.users)+1}", feature)
                        logger.info("已添加新用户到缓存")

            # 非循环模式且播放完毕则退出
            if not args.loop and camera.current_index == 0 and camera.frame_count > 1:
                logger.info("播放完毕")
                break

    except KeyboardInterrupt:
        logger.info("收到中断信号")

    finally:
        # 保存用户缓存
        if args.save_cache and cache.users:
            try:
                data = {}
                for user_id, user_data in cache.users.items():
                    data[str(user_id)] = {
                        'name': user_data['name'],
                        'embedding': user_data['embedding'].tolist()
                    }
                with open(args.save_cache, 'w') as f:
                    json.dump(data, f)
                logger.info(f"保存用户缓存到: {args.save_cache}")
            except Exception as e:
                logger.error(f"保存用户缓存失败: {e}")

        if not args.no_gui:
            cv2.destroyAllWindows()

        logger.info("测试结束")


if __name__ == "__main__":
    main()
