"""
MQTT 客户端模块
处理与后端的 MQTT 通信
"""
import json
import logging
import threading
from typing import Callable, Optional
import time

logger = logging.getLogger(__name__)

try:
    import paho.mqtt.client as mqtt
    MQTT_AVAILABLE = True
except ImportError:
    MQTT_AVAILABLE = False
    logger.warning("paho-mqtt 未安装，MQTT 功能将不可用")


class MQTTClient:
    """MQTT 客户端封装"""

    def __init__(
        self,
        broker: str,
        port: int,
        client_id: str,
        username: str = None,
        password: str = None
    ):
        """
        初始化 MQTT 客户端

        Args:
            broker: MQTT 代理地址
            port: MQTT 代理端口
            client_id: 客户端 ID
            username: 用户名（可选）
            password: 密码（可选）
        """
        if not MQTT_AVAILABLE:
            raise RuntimeError("paho-mqtt 库未安装")

        self.broker = broker
        self.port = port
        self.client_id = client_id
        self.username = username
        self.password = password

        self.client = mqtt.Client(client_id=client_id)
        self.connected = False
        self.callbacks = {}  # 主题 -> 回调函数映射

        # 设置认证信息
        if username and password:
            self.client.username_pw_set(username, password)

        # 设置回调
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_message = self._on_message

        logger.info(f"MQTT 客户端初始化: {client_id}")

    def _on_connect(self, client, userdata, flags, rc):
        """连接回调"""
        if rc == 0:
            self.connected = True
            logger.info(f"MQTT 连接成功: {self.broker}:{self.port}")

            # 重新订阅主题
            for topic in self.callbacks.keys():
                self.client.subscribe(topic)
                logger.info(f"订阅主题: {topic}")
        else:
            logger.error(f"MQTT 连接失败，返回码: {rc}")

    def _on_disconnect(self, client, userdata, rc):
        """断开连接回调"""
        self.connected = False
        if rc != 0:
            logger.warning(f"MQTT 意外断开，返回码: {rc}")

    def _on_message(self, client, userdata, msg):
        """消息回调"""
        try:
            topic = msg.topic
            payload = msg.payload.decode('utf-8')

            logger.debug(f"收到消息 [{topic}]: {payload}")

            # 调用对应的回调函数
            if topic in self.callbacks:
                callback = self.callbacks[topic]
                try:
                    # 尝试解析 JSON
                    try:
                        data = json.loads(payload)
                    except json.JSONDecodeError:
                        data = payload

                    callback(topic, data)
                except Exception as e:
                    logger.error(f"回调函数执行错误: {e}")

        except Exception as e:
            logger.error(f"消息处理错误: {e}")

    def connect(self) -> bool:
        """
        连接到 MQTT 代理

        Returns:
            是否连接成功
        """
        try:
            self.client.connect(self.broker, self.port, keepalive=60)
            self.client.loop_start()

            # 等待连接
            for _ in range(10):
                if self.connected:
                    return True
                time.sleep(0.5)

            logger.error("MQTT 连接超时")
            return False

        except Exception as e:
            logger.error(f"MQTT 连接异常: {e}")
            return False

    def disconnect(self):
        """断开连接"""
        try:
            self.client.loop_stop()
            self.client.disconnect()
            self.connected = False
            logger.info("MQTT 已断开连接")
        except Exception as e:
            logger.error(f"MQTT 断开连接异常: {e}")

    def subscribe(self, topic: str, callback: Callable):
        """
        订阅主题

        Args:
            topic: 主题名称
            callback: 回调函数 callback(topic, message)
        """
        self.callbacks[topic] = callback

        if self.connected:
            self.client.subscribe(topic)
            logger.info(f"订阅主题: {topic}")

    def unsubscribe(self, topic: str):
        """
        取消订阅主题

        Args:
            topic: 主题名称
        """
        if topic in self.callbacks:
            del self.callbacks[topic]

        if self.connected:
            self.client.unsubscribe(topic)
            logger.info(f"取消订阅主题: {topic}")

    def publish(self, topic: str, message, qos: int = 0, retain: bool = False):
        """
        发布消息

        Args:
            topic: 主题名称
            message: 消息内容（字典或字符串）
            qos: 服务质量等级
            retain: 是否保留消息
        """
        try:
            if isinstance(message, dict):
                payload = json.dumps(message)
            else:
                payload = str(message)

            self.client.publish(topic, payload, qos=qos, retain=retain)
            logger.debug(f"发布消息 [{topic}]: {payload}")

        except Exception as e:
            logger.error(f"发布消息失败: {e}")

    def is_connected(self) -> bool:
        """检查是否已连接"""
        return self.connected


class DoorControlMQTT:
    """门禁控制 MQTT 处理器"""

    def __init__(self, mqtt_client: MQTTClient, control_topic: str, status_topic: str):
        """
        初始化门禁控制处理器

        Args:
            mqtt_client: MQTT 客户端实例
            control_topic: 控制主题
            status_topic: 状态主题
        """
        self.mqtt_client = mqtt_client
        self.control_topic = control_topic
        self.status_topic = status_topic
        self.unlock_callback = None

        # 订阅控制主题
        self.mqtt_client.subscribe(control_topic, self._handle_control_message)

    def _handle_control_message(self, topic: str, message):
        """
        处理控制消息

        Args:
            topic: 主题
            message: 消息内容
        """
        logger.info(f"收到门禁控制指令: {message}")

        # 解析指令
        command = None
        if isinstance(message, dict):
            command = message.get("command", message.get("action", ""))
        else:
            command = str(message).upper()

        # 处理开门指令
        if command in ["UNLOCK", "OPEN", "UNLOCK_DOOR"]:
            logger.info("执行远程开门指令")
            if self.unlock_callback:
                try:
                    self.unlock_callback()
                except Exception as e:
                    logger.error(f"开门回调执行失败: {e}")

            # 发布状态
            self.publish_status("UNLOCKED", "Remote unlock via MQTT")

    def set_unlock_callback(self, callback: Callable):
        """
        设置开门回调函数

        Args:
            callback: 回调函数
        """
        self.unlock_callback = callback

    def publish_status(self, status: str, message: str = ""):
        """
        发布门禁状态

        Args:
            status: 状态 (LOCKED, UNLOCKED, ERROR)
            message: 状态消息
        """
        payload = {
            "status": status,
            "message": message,
            "timestamp": time.time()
        }
        self.mqtt_client.publish(self.status_topic, payload)

    def publish_heartbeat(self):
        """发布心跳消息"""
        self.publish_status("ONLINE", "Heartbeat")
