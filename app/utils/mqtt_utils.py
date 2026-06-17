import paho.mqtt.client as mqtt
import logging
import os
from typing import Optional, Callable

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MQTTClient:
    """MQTT客户端：用于与门禁设备通信"""

    def __init__(self):
        """初始化MQTT客户端"""
        self.client: Optional[mqtt.Client] = None
        self.broker = os.getenv("MQTT_BROKER", "localhost")
        self.port = int(os.getenv("MQTT_PORT", "1883"))
        self.username = os.getenv("MQTT_USERNAME", "")
        self.password = os.getenv("MQTT_PASSWORD", "")
        self.client_id = os.getenv("MQTT_CLIENT_ID", "door_access_server")
        self.connected = False
        self._initialize_client()

    def _initialize_client(self):
        """初始化MQTT客户端"""
        try:
            # 创建MQTT客户端
            self.client = mqtt.Client(client_id=self.client_id)

            # 设置回调函数
            self.client.on_connect = self._on_connect
            self.client.on_disconnect = self._on_disconnect
            self.client.on_publish = self._on_publish
            self.client.on_message = self._on_message

            # 设置用户名和密码（如果提供）
            if self.username and self.password:
                self.client.username_pw_set(self.username, self.password)

            logger.info("MQTT客户端初始化成功")

        except Exception as e:
            logger.error(f"MQTT客户端初始化失败: {e}")
            raise

    def connect(self) -> bool:
        """
        连接到MQTT代理

        Returns:
            连接是否成功
        """
        try:
            self.client.connect(self.broker, self.port, keepalive=60)
            self.client.loop_start()

            # 等待连接完成
            import time
            for _ in range(10):  # 最多等待5秒
                if self.connected:
                    logger.info(f"成功连接到MQTT代理: {self.broker}:{self.port}")
                    return True
                time.sleep(0.5)

            logger.warning("MQTT连接超时")
            return False

        except Exception as e:
            logger.error(f"MQTT连接失败: {e}")
            return False

    def disconnect(self):
        """断开MQTT连接"""
        try:
            if self.client and self.connected:
                self.client.loop_stop()
                self.client.disconnect()
                self.connected = False
                logger.info("MQTT连接已断开")
        except Exception as e:
            logger.error(f"MQTT断开连接失败: {e}")

    def publish(self, topic: str, payload: str, qos: int = 0, retain: bool = False) -> bool:
        """
        发布消息到指定主题

        Args:
            topic: 主题名称
            payload: 消息内容
            qos: 服务质量等级（0, 1, 2）
            retain: 是否保留消息

        Returns:
            发布是否成功
        """
        try:
            if not self.connected:
                logger.warning("MQTT未连接，无法发布消息")
                return False

            result = self.client.publish(topic, payload, qos=qos, retain=retain)

            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                logger.info(f"成功发布消息到主题 '{topic}': {payload}")
                return True
            else:
                logger.error(f"发布消息失败，错误代码: {result.rc}")
                return False

        except Exception as e:
            logger.error(f"发布消息异常: {e}")
            return False

    def subscribe(self, topic: str, qos: int = 0, callback: Optional[Callable] = None) -> bool:
        """
        订阅主题

        Args:
            topic: 主题名称
            qos: 服务质量等级
            callback: 消息回调函数

        Returns:
            订阅是否成功
        """
        try:
            if not self.connected:
                logger.warning("MQTT未连接，无法订阅主题")
                return False

            result = self.client.subscribe(topic, qos=qos)

            if result[0] == mqtt.MQTT_ERR_SUCCESS:
                logger.info(f"成功订阅主题: {topic}")
                if callback:
                    self.client.message_callback_add(topic, callback)
                return True
            else:
                logger.error(f"订阅主题失败，错误代码: {result[0]}")
                return False

        except Exception as e:
            logger.error(f"订阅主题异常: {e}")
            return False

    def send_door_command(self, command: str) -> bool:
        """
        发送门禁控制命令

        Args:
            command: 命令内容（如 "OPEN", "CLOSE"）

        Returns:
            发送是否成功
        """
        topic = "door/control"
        return self.publish(topic, command)

    # 回调函数
    def _on_connect(self, client, userdata, flags, rc):
        """连接回调"""
        if rc == 0:
            self.connected = True
            logger.info("MQTT客户端已连接")
        else:
            self.connected = False
            logger.error(f"MQTT连接失败，返回码: {rc}")

    def _on_disconnect(self, client, userdata, rc):
        """断开连接回调"""
        self.connected = False
        if rc != 0:
            logger.warning(f"MQTT意外断开连接，返回码: {rc}")
        else:
            logger.info("MQTT客户端已断开连接")

    def _on_publish(self, client, userdata, mid):
        """发布回调"""
        logger.debug(f"消息已发布，消息ID: {mid}")

    def _on_message(self, client, userdata, msg):
        """消息接收回调"""
        logger.info(f"收到消息 - 主题: {msg.topic}, 内容: {msg.payload.decode()}")


# 创建全局MQTT客户端实例
mqtt_client = MQTTClient()