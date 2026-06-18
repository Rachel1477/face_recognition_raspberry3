"""
树莓派人脸识别门禁系统 - 瘦身版
仅保留：图像采集、网络传输、接收指令执行 GPIO 开门
人脸识别逻辑迁移到后端 PC
"""
import os
import sys
import json
import time
import logging
import threading
from datetime import datetime
from queue import Queue
from typing import Optional

import cv2
import requests
import RPi.GPIO as GPIO

from config import (
    BACKEND_URL, CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT, FPS,
    MQTT_BROKER, MQTT_PORT, MQTT_USERNAME, MQTT_PASSWORD,
    MQTT_CLIENT_ID, MQTT_TOPIC_CONTROL, MQTT_TOPIC_STATUS,
    DOOR_RELAY_PIN, DOOR_UNLOCK_DURATION, API_TIMEOUT,
    RECOGNIZE_INTERVAL, IMAGE_QUALITY, BUTTON_PIN
)
from mqtt_client import MQTTClient, DoorControlMQTT
from gpio_controller import create_gpio_controller

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PiDoorSystem:
    """
    树莓派门禁系统（瘦身版）
    
    职责：
    1. 视频采集
    2. 图片压缩
    3. 发送到后端识别
    4. 接收后端指令控制 GPIO 开门
    """
    
    def __init__(self):
        logger.info("初始化树莓派门禁系统...")
        
        # GPIO 控制
        self.gpio_controller = create_gpio_controller(DOOR_RELAY_PIN, DOOR_UNLOCK_DURATION)
        
        # 初始化按键 GPIO
        self.button_pin = BUTTON_PIN
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.button_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        logger.info(f"按键初始化完成: GPIO {self.button_pin}")
        
        # MQTT
        self.mqtt_client = None
        self.door_mqtt = None
        self._init_mqtt()
        
        # 状态
        self.running = False
        self.cap = None
        
        # 控制标志
        self.last_recognize_time = 0
        
        logger.info("系统初始化完成")

    def _init_mqtt(self):
        """初始化 MQTT"""
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
                logger.warning("MQTT 连接失败")
                
        except Exception as e:
            logger.error(f"MQTT 初始化失败: {e}")
            self.mqtt_client = None

    def _remote_unlock(self):
        """远程开门回调"""
        logger.info("收到远程开门指令")
        self.gpio_controller.unlock()
        self.door_mqtt.publish_status("UNLOCKED", "Remote unlock via MQTT")

    def _compress_image(self, frame) -> bytes:
        """
        压缩图片以减少网络传输
        
        Args:
            frame: 原始图像
            
        Returns:
            压缩后的 JPEG 字节数据
        """
        # 可选：调整尺寸
        # target_width = 640
        # if frame.shape[1] > target_width:
        #     scale = target_width / frame.shape[1]
        #     frame = cv2.resize(frame, None, fx=scale, fy=scale)
        
        # 压缩为 JPEG
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), IMAGE_QUALITY]
        _, buffer = cv2.imencode('.jpg', frame, encode_param)
        return buffer.tobytes()

    def _send_recognize_request(self, image_bytes: bytes):
        """
        发送识别请求到后端（异步）
        
        Args:
            image_bytes: 压缩后的图片字节
        """
        def _do_request():
            try:
                start_time = time.time()
                
                files = {'image': ('frame.jpg', image_bytes, 'image/jpeg')}
                
                response = requests.post(
                    f"{BACKEND_URL}/identify/with-log",
                    files=files,
                    timeout=API_TIMEOUT
                )
                
                elapsed = (time.time() - start_time) * 1000
                
                if response.status_code == 200:
                    result = response.json()
                    
                    if result.get('access_granted'):
                        user_name = result.get('name', 'Unknown')
                        confidence = result.get('confidence', 0)
                        logger.info(f"✅ 识别通过: {user_name} (置信度: {confidence:.4f}, 耗时: {elapsed:.0f}ms)")
                        self.gpio_controller.unlock()
                    else:
                        logger.info(f"❌ 识别失败: {result.get('message', 'Unknown')} (耗时: {elapsed:.0f}ms)")
                        
                else:
                    logger.error(f"识别请求失败: HTTP {response.status_code}")
                    
            except requests.Timeout:
                logger.error("识别请求超时")
            except requests.ConnectionError:
                logger.error("无法连接到后端服务器")
            except Exception as e:
                logger.error(f"识别请求异常: {e}")
        
        # 在后台线程中执行网络请求
        thread = threading.Thread(target=_do_request, daemon=True)
        thread.start()

    def capture_loop(self):
        """
        视频捕获主循环（按键触发识别）
        """
        logger.info("启动视频捕获循环（按键触发模式）")
        
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, FPS)
        
        if not self.cap.isOpened():
            logger.error("无法打开摄像头")
            return
        
        try:
            while self.running:
                ret, frame = self.cap.read()
                
                if not ret:
                    logger.warning("读取摄像头帧失败")
                    time.sleep(0.1)
                    continue
                
                # 检测按键是否按下（GPIO.LOW 表示按下）
                if GPIO.input(self.button_pin) == GPIO.LOW:
                    logger.info("检测到按键按下，开始识别...")
                    
                    # 连续抓几帧，直到拿到一张清晰的图
                    for _ in range(5):
                        ret, frame = self.cap.read()
                        if ret:
                            break
                    
                    if ret:
                        # 压缩并发送识别请求
                        image_bytes = self._compress_image(frame)
                        self._send_recognize_request(image_bytes)
                        
                        # 防抖/冷却：识别后等 2 秒，防止按一下触发很多次
                        time.sleep(2.0)
                
                # 为了不让 CPU 100% 占用，即使不采集也要小睡一下
                time.sleep(0.05)
                    
        finally:
            if self.cap:
                self.cap.release()
            logger.info("视频捕获循环结束")

    def start(self):
        """启动系统"""
        if self.running:
            logger.warning("系统已在运行")
            return
        
        logger.info("启动树莓派门禁系统...")
        self.running = True
        
        # 启动视频捕获循环
        capture_thread = threading.Thread(target=self.capture_loop, daemon=True)
        capture_thread.start()
        
        logger.info("系统已启动")
        logger.info(f"后端地址: {BACKEND_URL}")
        logger.info(f"识别间隔: 每 {RECOGNIZE_INTERVAL} 帧")

    def stop(self):
        """停止系统"""
        if not self.running:
            return
        
        logger.info("停止系统...")
        self.running = False
        
        if self.cap:
            self.cap.release()
        
        if self.mqtt_client:
            self.mqtt_client.disconnect()
        
        logger.info("系统已停止")


def main():
    """主函数"""
    print("=" * 60)
    print("树莓派门禁系统 - 瘦身版")
    print("=" * 60)
    print("人脸识别由后端 PC 处理")
    print("-" * 60)
    
    system = PiDoorSystem()
    
    try:
        system.start()
        
        # 保持主线程运行
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n收到停止信号...")
    finally:
        system.stop()


if __name__ == "__main__":
    main()
