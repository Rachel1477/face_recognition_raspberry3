"""
树莓派人脸识别门禁系统 - 双因子验证版
保留：图像采集、网络传输、接收指令执行 GPIO 开门
新增：声纹识别验证
人脸识别逻辑迁移到后端 PC
"""
import os
import sys
import json
import time
import logging
import threading
import subprocess
from datetime import datetime
from queue import Queue
from typing import Optional

import cv2
import requests
import RPi.GPIO as GPIO
import numpy as np
from scipy.io import wavfile

from config import (
    BACKEND_URL, CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT, FPS,
    MQTT_BROKER, MQTT_PORT, MQTT_USERNAME, MQTT_PASSWORD,
    MQTT_CLIENT_ID, MQTT_TOPIC_CONTROL, MQTT_TOPIC_STATUS,
    DOOR_RELAY_PIN, DOOR_UNLOCK_DURATION, API_TIMEOUT,
    RECOGNIZE_INTERVAL, IMAGE_QUALITY, BUTTON_PIN,
    VOICE_SAMPLE_RATE, VOICE_RECORD_DURATION, VOICE_CHANNELS, VOICE_TEMP_FILE
)
from mqtt_client import MQTTClient, DoorControlMQTT
from gpio_controller import create_gpio_controller
from oled_display import OLEDDisplay

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PiDoorSystem:
    """
    树莓派门禁系统（双因子验证版）

    职责：
    1. 视频采集
    2. 图片压缩
    3. 发送到后端人脸识别
    4. 声纹采集与验证
    5. 双因子均通过后控制 GPIO 开门
    """

    def __init__(self):
        logger.info("初始化树莓派门禁系统...")

        # 测试 arecord 命令是否可用
        try:
            result = subprocess.run(
                ['arecord', '--version'],
                capture_output=True,
                text=True,
                timeout=5
            )
            logger.info(f"arecord 可用: {result.stdout.split()[2] if result.stdout else 'unknown'}")
        except Exception as e:
            logger.warning(f"arecord 测试失败: {e}")

        # GPIO 控制
        self.gpio_controller = create_gpio_controller(DOOR_RELAY_PIN, DOOR_UNLOCK_DURATION)

        # 初始化按键 GPIO
        self.button_pin = BUTTON_PIN
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.button_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        logger.info(f"按键初始化完成: GPIO {self.button_pin}")

        # OLED 屏幕
        self.oled_display = OLEDDisplay()

        # MQTT
        self.mqtt_client = None
        self.door_mqtt = None
        self._init_mqtt()

        # 状态
        self.running = False
        self.cap = None

        # 控制标志
        self.last_recognize_time = 0

        # 显示初始状态
        self.oled_display.show_idle()

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
        # 压缩为 JPEG
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), IMAGE_QUALITY]
        _, buffer = cv2.imencode('.jpg', frame, encode_param)
        return buffer.tobytes()

    def _record_audio(self, duration: int = VOICE_RECORD_DURATION) -> Optional[str]:
        """
        录制音频并保存为 WAV 文件（使用 arecord 命令）

        Args:
            duration: 录音时长（秒）

        Returns:
            录音文件路径，失败返回 None
        """
        try:
            logger.info(f"开始录音（arecord），时长: {duration}秒...")

            # 修改点：
            # 1. -D 使用 'plughw:2,0' 绕过系统损坏的 default 配置
            # 2. -f 改为使用具体参数，配合 plughw 进行自动重采样
            cmd = [
                'arecord',
                '-D', 'plughw:2,0',    # 强制指向你的 USB 麦克风
                '-d', str(duration),
                '-f', 'S16_LE',        # 使用 16位 小端格式
                '-r', '16000',         # 采样率 16000 (声纹识别标准)
                '-c', '1',             # 单声道
                VOICE_TEMP_FILE
            ]

            logger.info(f"执行命令: {' '.join(cmd)}")

            # 执行录音命令
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=duration + 5  # 留一点余量
            )

            if result.returncode != 0:
                logger.error(f"录音失败: {result.stderr}")
                return None

            # 验证文件是否生成
            if not os.path.exists(VOICE_TEMP_FILE):
                logger.error(f"录音文件未生成: {VOICE_TEMP_FILE}")
                return None

            # 检查文件大小（太小的文件可能录音失败）
            file_size = os.path.getsize(VOICE_TEMP_FILE)
            if file_size < 1000:  # 小于1KB可能是空文件
                logger.error(f"录音文件太小 ({file_size} bytes)，可能录音失败")
                return None

            logger.info(f"录音完成，保存至: {VOICE_TEMP_FILE} ({file_size} bytes)")
            return VOICE_TEMP_FILE

        except subprocess.TimeoutExpired:
            logger.error("录音超时")
            return None
        except Exception as e:
            logger.error(f"录音失败: {e}")
            return None

    def _send_voice_verify_request(self, user_id: int, audio_path: str) -> dict:
        """
        发送声纹验证请求到后端

        Args:
            user_id: 用户ID
            audio_path: 音频文件路径

        Returns:
            后端返回的验证结果
        """
        try:
            with open(audio_path, 'rb') as f:
                files = {'audio': ('voice.wav', f, 'audio/wav')}

                response = requests.post(
                    f"{BACKEND_URL}/identify/voice-verify",
                    files=files,
                    params={'user_id': user_id},
                    timeout=API_TIMEOUT
                )

            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"声纹验证请求失败: HTTP {response.status_code}")
                return {'access_granted': False, 'message': f'HTTP {response.status_code}'}

        except requests.Timeout:
            logger.error("声纹验证请求超时")
            return {'access_granted': False, 'message': '请求超时'}
        except requests.ConnectionError:
            logger.error("无法连接到后端服务器")
            return {'access_granted': False, 'message': '连接失败'}
        except Exception as e:
            logger.error(f"声纹验证请求异常: {e}")
            return {'access_granted': False, 'message': str(e)}

    def _handle_face_result(self, result: dict):
        """
        处理人脸识别结果，根据状态决定是否需要声纹验证

        Args:
            result: 人脸识别返回结果
        """
        user_name = result.get('name', 'Unknown')
        user_id = result.get('user_id')
        confidence = result.get('confidence', 0)
        status = result.get('status', 'denied')

        if status == 'need_voice':
            # 人脸通过，需要声纹验证
            logger.info(f"人脸验证通过: {user_name}，进入声纹验证...")

            # OLED 显示请说话
            self.oled_display.show_please_speak(8)

            # 录制音频
            audio_path = self._record_audio(VOICE_RECORD_DURATION)

            if audio_path is None:
                logger.error("录音失败")
                self.oled_display.show_face_pass_voice_fail()
                return

            # OLED 显示语音识别中
            self.oled_display.show_voice_recognizing()

            # 发送声纹验证请求
            voice_result = self._send_voice_verify_request(user_id, audio_path)

            if voice_result.get('access_granted'):
                # 声纹也通过了，允许开门
                voice_confidence = voice_result.get('confidence', 0)
                logger.info(f"声纹验证通过: {user_name} (置信度: {voice_confidence:.4f})")
                self.gpio_controller.unlock()
                self.oled_display.show_voice_pass(user_name)
            else:
                # 声纹验证失败
                voice_confidence = voice_result.get('confidence', 0)
                logger.info(f"声纹验证失败: {user_name} (置信度: {voice_confidence:.4f})")
                self.oled_display.show_face_pass_voice_fail()

        elif status == 'granted':
            # 预留：未来可能需要的直接通过场景
            logger.info(f"识别通过: {user_name} (置信度: {confidence:.4f})")
            self.gpio_controller.unlock()
            self.oled_display.show_success(user_name)
        else:
            # 人脸验证失败
            logger.info(f"识别失败: {result.get('message', 'Unknown')}")
            self.oled_display.show_failed()

    def _send_recognize_request(self, image_bytes: bytes):
        """
        发送识别请求到后端（异步）

        Args:
            image_bytes: 压缩后的图片字节
        """
        # 更新屏幕状态：正在识别
        self.oled_display.show_recognizing()

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
                    logger.info(f"识别结果: {result.get('status')} (耗时: {elapsed:.0f}ms)")

                    # 处理识别结果（含声纹验证逻辑）
                    self._handle_face_result(result)
                else:
                    logger.error(f"识别请求失败: HTTP {response.status_code}")
                    self.oled_display.show_failed()

            except requests.Timeout:
                logger.error("识别请求超时")
                self.oled_display.show_failed()
            except requests.ConnectionError:
                logger.error("无法连接到后端服务器")
                self.oled_display.show_failed()
            except Exception as e:
                logger.error(f"识别请求异常: {e}")
                self.oled_display.show_failed()

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

                    # 更新屏幕状态：采集成功
                    self.oled_display.show_captured()

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
                        # 这段时间屏幕会显示成功/失败状态
                        time.sleep(2.0)

                        # 重置屏幕状态：待激活
                        self.oled_display.show_idle()

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

        if self.oled_display:
            self.oled_display.cleanup()

        logger.info("系统已停止")


def main():
    """主函数"""
    print("=" * 60)
    print("树莓派门禁系统 - 双因子验证版")
    print("=" * 60)
    print("人脸识别 + 声纹识别")
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
