"""
配置文件 - 树莓派人脸识别系统
使用 OpenCV 官方人脸识别接口
"""
import os

# 后端服务器配置
BACKEND_URL = os.getenv("BACKEND_URL", "http://192.168.160.11:8000")
API_TIMEOUT = 10

# 摄像头配置
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 15
FRAME_SKIP = 3  # 每隔 N 帧处理一次

# OpenCV Face 模块配置
FACE_DETECTOR_MODEL = "face_detection_yunet_2023mar.onnx"
FACE_RECOGNIZER_MODEL = "face_recognition_sface_2021dec.onnx"

# 识别配置
COSINE_THRESHOLD = 0.36  # SFace 推荐阈值

# GPIO 配置
DOOR_RELAY_PIN = 18
DOOR_UNLOCK_DURATION = 3

# MQTT 配置
MQTT_BROKER = os.getenv("MQTT_BROKER", "192.168.160.11")
MQTT_PORT = int(os.getenv("MQTT_PORT", 1883))
MQTT_USERNAME = os.getenv("MQTT_USERNAME", "")
MQTT_PASSWORD = os.getenv("MQTT_PASSWORD", "")
MQTT_CLIENT_ID = "raspberry_pi_door"
MQTT_TOPIC_CONTROL = "door/control"
MQTT_TOPIC_STATUS = "door/status"

# 用户缓存配置
USER_SYNC_INTERVAL = 300
LOCAL_CACHE_FILE = "user_cache.json"

# 日志配置
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# 抓拍配置
CAPTURE_DIR = "captures"
MAX_CAPTURE_FILES = 100
