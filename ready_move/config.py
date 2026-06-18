"""
配置文件 - 树莓派门禁系统（瘦身版）
人脸识别逻辑迁移到后端 PC
"""
import os

# 后端服务器配置
BACKEND_URL = os.getenv("BACKEND_URL", "http://192.168.160.11:8000")
API_TIMEOUT = int(os.getenv("API_TIMEOUT", 10))

# 摄像头配置
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 15

# 识别配置（树莓派端）
RECOGNIZE_INTERVAL = 15  # 每隔 N 帧发送一次识别请求（约 1 秒）
IMAGE_QUALITY = 85  # JPEG 压缩质量 (0-100)

# GPIO 配置
DOOR_RELAY_PIN = int(os.getenv("DOOR_RELAY_PIN", 18))
DOOR_UNLOCK_DURATION = int(os.getenv("DOOR_UNLOCK_DURATION", 3))

# MQTT 配置
MQTT_BROKER = os.getenv("MQTT_BROKER", "192.168.160.11")
MQTT_PORT = int(os.getenv("MQTT_PORT", 1883))
MQTT_USERNAME = os.getenv("MQTT_USERNAME", "")
MQTT_PASSWORD = os.getenv("MQTT_PASSWORD", "")
MQTT_CLIENT_ID = "raspberry_pi_door"
MQTT_TOPIC_CONTROL = "door/control"
MQTT_TOPIC_STATUS = "door/status"

# 日志配置
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
