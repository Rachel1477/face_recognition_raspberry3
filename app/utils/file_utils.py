"""
文件工具模块
处理图片保存等文件操作
"""
import os
import uuid
import cv2
import numpy as np
from datetime import datetime
from typing import Optional

# 访问图片保存目录
ACCESS_IMAGE_DIR = "static/access_images"


async def save_access_image(image: np.ndarray) -> str:
    """
    保存访问抓拍图片
    
    Args:
        image: BGR 格式的图像
        
    Returns:
        保存后的相对路径
    """
    # 确保目录存在
    os.makedirs(ACCESS_IMAGE_DIR, exist_ok=True)
    
    # 生成文件名
    filename = f"{uuid.uuid4().hex}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    image_path = os.path.join(ACCESS_IMAGE_DIR, filename)
    
    # 保存图片
    cv2.imwrite(image_path, image)
    
    return image_path


def save_base64_image(base64_data: str, subdir: str = "") -> Optional[str]:
    """
    保存 Base64 编码的图片
    
    Args:
        base64_data: Base64 编码的图片数据
        subdir: 子目录
        
    Returns:
        保存后的相对路径，或 None（失败时）
    """
    try:
        import base64
        import os
        
        # 目录
        save_dir = ACCESS_IMAGE_DIR
        if subdir:
            save_dir = os.path.join(ACCESS_IMAGE_DIR, subdir)
        
        os.makedirs(save_dir, exist_ok=True)
        
        # 解码
        if "," in base64_data:
            base64_data = base64_data.split(",")[1]
        
        image_bytes = base64.b64decode(base64_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return None
        
        # 生成文件名
        filename = f"{uuid.uuid4().hex}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        image_path = os.path.join(save_dir, filename)
        
        # 保存
        cv2.imwrite(image_path, img)
        
        return image_path
        
    except Exception as e:
        print(f"保存 Base64 图片失败: {e}")
        return None
