"""
后端配置文件
"""
import os

# 项目根目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 模型目录
MODEL_DIR = os.path.join(BASE_DIR, "app", "models")

# 人脸识别配置
COSINE_THRESHOLD = 0.5  # 余弦相似度阈值（insightface buffalo_l 使用 0.5）

# 声纹识别配置
VOICE_THRESHOLD = 0.6  # 声纹相似度阈值

# 静态文件目录
STATIC_DIR = os.path.join(BASE_DIR, "static")
FACES_DIR = os.path.join(STATIC_DIR, "faces")
ACCESS_IMAGES_DIR = os.path.join(STATIC_DIR, "access_images")
VOICE_DIR = os.path.join(STATIC_DIR, "voices")
