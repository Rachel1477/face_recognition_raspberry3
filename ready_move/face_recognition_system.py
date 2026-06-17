"""
树莓派人脸识别门禁系统 - 主程序
使用 OpenCV 官方人脸识别接口：
- cv2.FaceDetectorYN: 人脸检测
- cv2.FaceRecognizerSF: 人脸识别
"""
import os
import sys
import json
import time
import logging
import threading
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from queue import Queue
import base64

import cv2
import numpy as np
import requests

from config import *
from face_embedding import FaceRecognizer
from mqtt_client import MQTTClient, DoorControlMQTT
from gpio_controller import create_gpio_controller

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format=LOG_FORMAT
)
logger = logging.getLogger(__name__)


class UserCache:
    """用户缓存管理"""

    def __init__(self, cache_file: str = LOCAL_CACHE_FILE):
        self.cache_file = cache_file
        self.users: Dict[int, dict] = {}
        self._lock = threading.Lock()
        self._load_cache()

    def _load_cache(self):
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    data = json.load(f)
                    for user_id, user_data in data.items():
                        user_data['embedding'] = np.array(user_data['embedding'])
                        self.users[int(user_id)] = user_data
                logger.info(f"加载本地用户缓存: {len(self.users)} 个用户")
            except Exception as e:
                logger.error(f"加载本地缓存失败: {e}")

    def _save_cache(self):
        try:
            data = {}
            for user_id, user_data in self.users.items():
                data[str(user_id)] = {
                    'name': user_data['name'],
                    'embedding': user_data['embedding'].tolist()
                }
            with open(self.cache_file, 'w') as f:
                json.dump(data, f)
            logger.info(f"保存用户缓存到本地: {len(self.users)} 个用户")
        except Exception as e:
            logger.error(f"保存本地缓存失败: {e}")

    def update_users(self, users: List[dict]):
        with self._lock:
            self.users.clear()
            for user in users:
                if user.get('face_vector'):
                    try:
                        embedding = np.array(json.loads(user['face_vector']))
                        self.users[user['id']] = {
                            'name': user['name'],
                            'embedding': embedding
                        }
                    except Exception as e:
                        logger.error(f"解析用户 {user['id']} 特征向量失败: {e}")
            self._save_cache()
            logger.info(f"更新用户缓存: {len(self.users)} 个用户")

    def find_best_match(self, embedding: np.ndarray, threshold: float = COSINE_THRESHOLD) -> Optional[Tuple[int, str, float]]:
        with self._lock:
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


class BackendAPI:
    """后端 API 客户端"""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.timeout = API_TIMEOUT

    def get_users(self) -> List[dict]:
        try:
            response = self.session.get(f"{self.base_url}/users")
            response.raise_for_status()
            data = response.json()
            return data.get('users', [])
        except Exception as e:
            logger.error(f"获取用户列表失败: {e}")
            return []

    def create_access_log(self, user_id: Optional[int], status: str,
                          image_path: str, confidence: Optional[float] = None) -> bool:
        try:
            with open(image_path, 'rb') as f:
                image_data = base64.b64encode(f.read()).decode()

            payload = {
                "user_id": user_id,
                "status": status,
                "image_data": image_data,
                "confidence": str(confidence) if confidence else None
            }

            response = self.session.post(
                f"{self.base_url}/logs/log",
                json=payload
            )
            response.raise_for_status()
            logger.info(f"访问日志上传成功: {status}")
            return True

        except Exception as e:
            logger.error(f"创建访问日志失败: {e}")
            return False


class FaceRecognitionSystem:
    """人脸识别门禁系统"""

    def __init__(self):
        logger.info("初始化人脸识别门禁系统...")

        self.user_cache = UserCache()
        self.backend_api = BackendAPI(BACKEND_URL)
        self.gpio_controller = create_gpio_controller(DOOR_RELAY_PIN, DOOR_UNLOCK_DURATION)

        # 初始化 OpenCV 人脸识别器
        self.face_recognizer = FaceRecognizer(
            FACE_DETECTOR_MODEL,
            FACE_RECOGNIZER_MODEL
        )

        # MQTT
        self.mqtt_client = None
        self.door_mqtt = None
        self._init_mqtt()

        self.running = False
        self.frame_queue = Queue(maxsize=4)
        self.recognition_queue = Queue(maxsize=4)
        self.result_queue = Queue(maxsize=10)

        os.makedirs(CAPTURE_DIR, exist_ok=True)
        self.sync_timer = None

        logger.info("系统初始化完成")

    def _init_mqtt(self):
        try:
            self.mqtt_client = MQTTClient(
                broker=MQTT_BROKER,
                port=MQTT_PORT,
                client_id=MQTT_CLIENT_ID,
                username=MQTT_USERNAME if MQTT_USERNAME else None,
                password=MQTT_PASSWORD if MQTT_PASSWORD else None
            )

            if self.mqtt_client.connect():
                self.door_mqtt = DoorControlMQTT(
                    self.mqtt_client,
                    MQTT_TOPIC_CONTROL,
                    MQTT_TOPIC_STATUS
                )
                self.door_mqtt.set_unlock_callback(self._remote_unlock)
                logger.info("MQTT 连接成功")
            else:
                logger.warning("MQTT 连接失败，远程控制功能不可用")

        except Exception as e:
            logger.error(f"MQTT 初始化失败: {e}")
            self.mqtt_client = None

    def _remote_unlock(self):
        logger.info("执行远程开门指令")
        self.gpio_controller.unlock()

    def sync_users(self):
        logger.info("同步用户数据...")
        users = self.backend_api.get_users()
        if users:
            self.user_cache.update_users(users)

        if self.running:
            self.sync_timer = threading.Timer(USER_SYNC_INTERVAL, self.sync_users)
            self.sync_timer.daemon = True
            self.sync_timer.start()

    def capture_thread(self):
        """视频捕获线程"""
        logger.info("启动视频捕获线程")

        cap = cv2.VideoCapture(CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, FPS)

        if not cap.isOpened():
            logger.error("无法打开摄像头")
            return

        frame_count = 0

        while self.running:
            ret, frame = cap.read()
            if not ret:
                logger.warning("读取摄像头帧失败")
                continue

            frame_count += 1

            # 每隔 FRAME_SKIP 帧处理一次
            if frame_count % FRAME_SKIP == 0:
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except:
                        pass
                self.frame_queue.put(frame.copy())

        cap.release()
        logger.info("视频捕获线程结束")

    def recognition_thread(self):
        """人脸识别线程（检测 + 识别）"""
        logger.info("启动人脸识别线程")

        last_recognition_time = 0
        recognition_interval = 2.0

        while self.running:
            try:
                frame = self.frame_queue.get(timeout=1.0)
            except:
                continue

            current_time = time.time()
            if current_time - last_recognition_time < recognition_interval:
                continue
            last_recognition_time = current_time

            # 检测并识别
            result = self.face_recognizer.recognize(frame)

            if result is not None:
                aligned_face, feature = result

                if self.recognition_queue.full():
                    try:
                        self.recognition_queue.get_nowait()
                    except:
                        pass
                self.recognition_queue.put((frame, feature))

        logger.info("人脸识别线程结束")

    def match_thread(self):
        """用户匹配线程"""
        logger.info("启动用户匹配线程")

        while self.running:
            try:
                frame, feature = self.recognition_queue.get(timeout=1.0)
            except:
                continue

            match_result = self.user_cache.find_best_match(feature)

            if match_result:
                user_id, name, similarity = match_result
                logger.info(f"识别成功: {name} (相似度: {similarity:.2f})")

                self.gpio_controller.unlock()

                capture_path = self._save_capture(frame, "success")

                self.result_queue.put({
                    'user_id': user_id,
                    'status': 'success',
                    'image_path': capture_path,
                    'confidence': similarity
                })
            else:
                logger.info("识别失败: 未知人员")

                capture_path = self._save_capture(frame, "failure")

                self.result_queue.put({
                    'user_id': None,
                    'status': 'failure',
                    'image_path': capture_path,
                    'confidence': None
                })

        logger.info("用户匹配线程结束")

    def upload_thread(self):
        """日志上传线程"""
        logger.info("启动日志上传线程")

        while self.running:
            try:
                result = self.result_queue.get(timeout=1.0)

                self.backend_api.create_access_log(
                    user_id=result['user_id'],
                    status=result['status'],
                    image_path=result['image_path'],
                    confidence=result['confidence']
                )

                self._cleanup_captures()

            except:
                continue

        logger.info("日志上传线程结束")

    def _save_capture(self, frame: np.ndarray, status: str) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{status}_{timestamp}.jpg"
        filepath = os.path.join(CAPTURE_DIR, filename)

        cv2.imwrite(filepath, frame)
        logger.debug(f"保存抓拍图片: {filepath}")

        return filepath

    def _cleanup_captures(self):
        try:
            files = sorted(
                [f for f in os.listdir(CAPTURE_DIR) if f.endswith('.jpg')],
                key=lambda x: os.path.getmtime(os.path.join(CAPTURE_DIR, x))
            )

            while len(files) > MAX_CAPTURE_FILES:
                old_file = os.path.join(CAPTURE_DIR, files.pop(0))
                os.remove(old_file)
                logger.debug(f"删除旧抓拍文件: {old_file}")

        except Exception as e:
            logger.error(f"清理抓拍文件失败: {e}")

    def run(self):
        logger.info("启动人脸识别门禁系统")
        self.running = True

        self.sync_users()

        threads = [
            threading.Thread(target=self.capture_thread, name="CaptureThread"),
            threading.Thread(target=self.recognition_thread, name="RecognitionThread"),
            threading.Thread(target=self.match_thread, name="MatchThread"),
            threading.Thread(target=self.upload_thread, name="UploadThread")
        ]

        for t in threads:
            t.daemon = True
            t.start()

        display_frame = None

        try:
            while self.running:
                try:
                    frame = self.frame_queue.get_nowait()
                    display_frame = frame.copy()
                except:
                    pass

                if display_frame is not None:
                    status = self.gpio_controller.get_status()
                    cv2.putText(display_frame, f"Door: {status}", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(display_frame, f"Users: {len(self.user_cache.users)}", (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

                    cv2.imshow("Face Recognition System", display_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except KeyboardInterrupt:
            logger.info("收到中断信号")

        finally:
            self.stop()

    def stop(self):
        logger.info("停止人脸识别门禁系统")
        self.running = False

        if self.sync_timer:
            self.sync_timer.cancel()

        if self.mqtt_client:
            self.mqtt_client.disconnect()

        self.gpio_controller.cleanup()

        cv2.destroyAllWindows()

        logger.info("系统已停止")


def main():
    if not os.path.exists(FACE_DETECTOR_MODEL):
        logger.error(f"检测器模型不存在: {FACE_DETECTOR_MODEL}")
        logger.info("请下载 face_detection_yunet_2023mar.onnx")
        sys.exit(1)

    if not os.path.exists(FACE_RECOGNIZER_MODEL):
        logger.error(f"识别器模型不存在: {FACE_RECOGNIZER_MODEL}")
        logger.info("请下载 face_recognition_sface_2021dec.onnx")
        sys.exit(1)

    system = FaceRecognitionSystem()

    try:
        system.run()
    except Exception as e:
        logger.error(f"系统运行错误: {e}")
        system.stop()


if __name__ == "__main__":
    main()
