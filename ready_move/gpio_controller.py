"""
GPIO 控制模块
控制树莓派 GPIO 引脚实现开门动作（舵机版本）
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

# 舵机配置
SERVO_FREQUENCY = 50  # 舵机 PWM 频率 (Hz)
SERVO_LOCKED_DUTY = 2.5  # 关门位置 (0°) 的占空比
SERVO_UNLOCKED_DUTY = 10.0  # 开门位置 (120°) 的占空比


class ServoGPIOController:
    """舵机 GPIO 控制器"""

    def __init__(self, servo_pin: int, unlock_duration: float = 3.0):
        """
        初始化舵机 GPIO 控制器

        Args:
            servo_pin: 舵机控制引脚（BCM编号）
            unlock_duration: 开门持续时间（秒）
        """
        self.servo_pin = servo_pin
        self.unlock_duration = unlock_duration
        self.is_unlocked = False
        self._lock = threading.Lock()
        self._pwm = None

        if GPIO_AVAILABLE:
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)
            GPIO.setup(servo_pin, GPIO.OUT)

            # 初始化 PWM，频率 50Hz
            self._pwm = GPIO.PWM(servo_pin, SERVO_FREQUENCY)
            self._pwm.start(SERVO_LOCKED_DUTY)

            logger.info(f"舵机 GPIO 初始化成功，引脚: {servo_pin}，初始位置: 关门 (0°)")
        else:
            logger.info(f"GPIO 模拟模式，舵机引脚: {servo_pin}")

    def _set_angle(self, duty_cycle: float):
        """设置舵机角度"""
        if GPIO_AVAILABLE and self._pwm:
            # 设置角度
            self._pwm.ChangeDutyCycle(duty_cycle)
            time.sleep(0.5)  # 等待舵机转动到位
            self._pwm.ChangeDutyCycle(0)  # 停止信号，防止抖动
        else:
            logger.info(f"[模拟] 设置舵机占空比: {duty_cycle}%")

    def unlock(self, duration: Optional[float] = None):
        """
        开门：顺时针旋转120°

        Args:
            duration: 开门持续时间（秒），None 使用默认值
        """
        with self._lock:
            if self.is_unlocked:
                logger.warning("门已处于解锁状态")
                return

            duration = duration or self.unlock_duration
            self.is_unlocked = True

            # 顺时针旋转120°（开门）
            self._set_angle(SERVO_UNLOCKED_DUTY)
            logger.info(f"开门（舵机顺时针旋转120°），持续 {duration} 秒")

            timer = threading.Timer(duration, self._auto_lock)
            timer.daemon = True
            timer.start()

    def _auto_lock(self):
        """自动关门：逆时针旋转120°复位"""
        with self._lock:
            if not self.is_unlocked:
                return

            self.is_unlocked = False

            # 逆时针旋转120°（关门复位）
            self._set_angle(SERVO_LOCKED_DUTY)
            logger.info("自动关门（舵机逆时针旋转120°复位）")

    def lock(self):
        """手动关门：逆时针旋转120°复位"""
        with self._lock:
            if not self.is_unlocked:
                logger.info("门已处于锁定状态")
                return

            self.is_unlocked = False

            # 逆时针旋转120°（关门复位）
            self._set_angle(SERVO_LOCKED_DUTY)
            logger.info("手动关门（舵机逆时针旋转120°复位）")

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
            if self._pwm:
                self._pwm.stop()
            GPIO.cleanup(self.servo_pin)
            logger.info("舵机 GPIO 资源已清理")


class MockServoController:
    """模拟舵机控制器（用于测试）"""

    def __init__(self, servo_pin: int, unlock_duration: float = 3.0):
        self.servo_pin = servo_pin
        self.unlock_duration = unlock_duration
        self.is_unlocked = False
        self._lock = threading.Lock()
        logger.info(f"模拟舵机控制器初始化，引脚: {servo_pin}")

    def unlock(self, duration: Optional[float] = None):
        duration = duration or self.unlock_duration
        with self._lock:
            if self.is_unlocked:
                return
            self.is_unlocked = True
            logger.info(f"[模拟] 开门（舵机顺时针旋转120°），持续 {duration} 秒")

            timer = threading.Timer(duration, self._auto_lock)
            timer.daemon = True
            timer.start()

    def _auto_lock(self):
        with self._lock:
            self.is_unlocked = False
            logger.info("[模拟] 自动关门（舵机逆时针旋转120°复位）")

    def lock(self):
        with self._lock:
            if not self.is_unlocked:
                return
            self.is_unlocked = False
            logger.info("[模拟] 手动关门（舵机逆时针旋转120°复位）")

    def get_status(self) -> str:
        return "UNLOCKED" if self.is_unlocked else "LOCKED"

    def cleanup(self):
        logger.info("[模拟] 舵机 GPIO 资源已清理")


def create_gpio_controller(servo_pin: int, unlock_duration: float = 3.0):
    """
    创建舵机控制器

    Args:
        servo_pin: 舵机控制引脚
        unlock_duration: 开门持续时间

    Returns:
        舵机控制器实例
    """
    if GPIO_AVAILABLE:
        return ServoGPIOController(servo_pin, unlock_duration)
    else:
        return MockServoController(servo_pin, unlock_duration)
