"""
GPIO 控制模块
控制树莓派 GPIO 引脚实现开门动作
"""
import logging
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)

# 尝试导入 GPIO 库
try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False
    logger.warning("RPi.GPIO 未安装，GPIO 功能将使用模拟模式")


class GPIOController:
    """GPIO 控制器"""

    def __init__(self, relay_pin: int, unlock_duration: float = 3.0):
        """
        初始化 GPIO 控制器

        Args:
            relay_pin: 继电器控制引脚
            unlock_duration: 开门持续时间（秒）
        """
        self.relay_pin = relay_pin
        self.unlock_duration = unlock_duration
        self.is_unlocked = False
        self._lock = threading.Lock()

        if GPIO_AVAILABLE:
            # 设置 GPIO 模式
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)

            # 设置引脚为输出
            GPIO.setup(relay_pin, GPIO.OUT)

            # 初始状态：锁定
            GPIO.output(relay_pin, GPIO.HIGH)

            logger.info(f"GPIO 初始化成功，继电器引脚: {relay_pin}")
        else:
            logger.info(f"GPIO 模拟模式，继电器引脚: {relay_pin}")

    def unlock(self, duration: Optional[float] = None):
        """
        开门

        Args:
            duration: 开门持续时间（秒），None 使用默认值
        """
        with self._lock:
            if self.is_unlocked:
                logger.warning("门已处于解锁状态")
                return

            duration = duration or self.unlock_duration
            self.is_unlocked = True

            if GPIO_AVAILABLE:
                # 触发继电器（低电平触发）
                GPIO.output(self.relay_pin, GPIO.LOW)
                logger.info(f"开门，持续 {duration} 秒")
            else:
                logger.info(f"[模拟] 开门，持续 {duration} 秒")

            # 启动定时器自动关门
            timer = threading.Timer(duration, self._auto_lock)
            timer.daemon = True
            timer.start()

    def _auto_lock(self):
        """自动关门"""
        with self._lock:
            if not self.is_unlocked:
                return

            self.is_unlocked = False

            if GPIO_AVAILABLE:
                GPIO.output(self.relay_pin, GPIO.HIGH)
                logger.info("自动关门")
            else:
                logger.info("[模拟] 自动关门")

    def lock(self):
        """手动关门"""
        with self._lock:
            if not self.is_unlocked:
                logger.info("门已处于锁定状态")
                return

            self.is_unlocked = False

            if GPIO_AVAILABLE:
                GPIO.output(self.relay_pin, GPIO.HIGH)
                logger.info("手动关门")
            else:
                logger.info("[模拟] 手动关门")

    def get_status(self) -> str:
        """
        获取门禁状态

        Returns:
            状态字符串 (LOCKED 或 UNLOCKED)
        """
        return "UNLOCKED" if self.is_unlocked else "LOCKED"

    def cleanup(self):
        """清理 GPIO 资源"""
        if GPIO_AVAILABLE:
            GPIO.cleanup(self.relay_pin)
            logger.info("GPIO 资源已清理")


class MockGPIOController:
    """模拟 GPIO 控制器（用于测试）"""

    def __init__(self, relay_pin: int, unlock_duration: float = 3.0):
        self.relay_pin = relay_pin
        self.unlock_duration = unlock_duration
        self.is_unlocked = False
        self._lock = threading.Lock()
        logger.info(f"模拟 GPIO 控制器初始化，引脚: {relay_pin}")

    def unlock(self, duration: Optional[float] = None):
        duration = duration or self.unlock_duration
        with self._lock:
            self.is_unlocked = True
            logger.info(f"[模拟] 开门，持续 {duration} 秒")

            timer = threading.Timer(duration, self._auto_lock)
            timer.daemon = True
            timer.start()

    def _auto_lock(self):
        with self._lock:
            self.is_unlocked = False
            logger.info("[模拟] 自动关门")

    def lock(self):
        with self._lock:
            self.is_unlocked = False
            logger.info("[模拟] 手动关门")

    def get_status(self) -> str:
        return "UNLOCKED" if self.is_unlocked else "LOCKED"

    def cleanup(self):
        logger.info("[模拟] GPIO 资源已清理")


def create_gpio_controller(relay_pin: int, unlock_duration: float = 3.0):
    """
    创建 GPIO 控制器

    Args:
        relay_pin: 继电器引脚
        unlock_duration: 开门持续时间

    Returns:
        GPIO 控制器实例
    """
    if GPIO_AVAILABLE:
        return GPIOController(relay_pin, unlock_duration)
    else:
        return MockGPIOController(relay_pin, unlock_duration)
