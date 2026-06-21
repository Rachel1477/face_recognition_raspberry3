"""
GPIO 控制模块
控制树莓派 GPIO 引脚实现开门动作（舵机版本）
包含 LED 灯控制
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
        self._on_door_closed_callback = None  # 关门回调

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

    def set_door_closed_callback(self, callback):
        """
        设置关门回调函数

        Args:
            callback: 关门时要调用的回调函数
        """
        self._on_door_closed_callback = callback

    def _notify_door_closed(self):
        """通知门已关闭"""
        if self._on_door_closed_callback:
            try:
                self._on_door_closed_callback()
            except Exception as e:
                logger.error(f"关门回调执行失败: {e}")

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

        # 通知门已关闭（回调）
        self._notify_door_closed()

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

        # 通知门已关闭（回调）
        self._notify_door_closed()

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
        self._on_door_closed_callback = None
        logger.info(f"模拟舵机控制器初始化，引脚: {servo_pin}")

    def set_door_closed_callback(self, callback):
        """设置关门回调函数"""
        self._on_door_closed_callback = callback

    def _notify_door_closed(self):
        """通知门已关闭"""
        if self._on_door_closed_callback:
            try:
                self._on_door_closed_callback()
            except Exception as e:
                logger.error(f"关门回调执行失败: {e}")

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
        self._notify_door_closed()

    def lock(self):
        with self._lock:
            if not self.is_unlocked:
                return
            self.is_unlocked = False
            logger.info("[模拟] 手动关门（舵机逆时针旋转120°复位）")
        self._notify_door_closed()

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


class LEDController:
    """LED 灯控制器"""

    def __init__(self, green_pin: int, blue_pin: int):
        """
        初始化 LED 控制器

        Args:
            green_pin: 绿灯引脚（BCM编号）- 开门时亮
            blue_pin: 蓝灯引脚（BCM编号）- 人脸识别通过后亮
        """
        self.green_pin = green_pin
        self.blue_pin = blue_pin
        self._green_on = False
        self._blue_on = False

        if GPIO_AVAILABLE:
            GPIO.setup(green_pin, GPIO.OUT)
            GPIO.setup(blue_pin, GPIO.OUT)
            # 初始状态：全部熄灭
            GPIO.output(green_pin, GPIO.LOW)
            GPIO.output(blue_pin, GPIO.LOW)
            logger.info(f"LED 初始化完成，绿灯: GPIO {green_pin}，蓝灯: GPIO {blue_pin}")
        else:
            logger.info(f"LED 模拟模式，绿灯: GPIO {green_pin}，蓝灯: GPIO {blue_pin}")

    def green_on(self):
        """打开绿灯"""
        self._green_on = True
        if GPIO_AVAILABLE:
            GPIO.output(self.green_pin, GPIO.HIGH)
        logger.info(f"绿灯亮 (GPIO {self.green_pin})")

    def green_off(self):
        """关闭绿灯"""
        self._green_on = False
        if GPIO_AVAILABLE:
            GPIO.output(self.green_pin, GPIO.LOW)
        logger.info(f"绿灯灭 (GPIO {self.green_pin})")

    def blue_on(self):
        """打开蓝灯"""
        self._blue_on = True
        if GPIO_AVAILABLE:
            GPIO.output(self.blue_pin, GPIO.HIGH)
        logger.info(f"蓝灯亮 (GPIO {self.blue_pin})")

    def blue_off(self):
        """关闭蓝灯"""
        self._blue_on = False
        if GPIO_AVAILABLE:
            GPIO.output(self.blue_pin, GPIO.LOW)
        logger.info(f"蓝灯灭 (GPIO {self.blue_pin})")

    def all_off(self):
        """关闭所有灯"""
        self.green_off()
        self.blue_off()

    def face_passed(self):
        """人脸识别通过：亮蓝灯"""
        self.all_off()
        self.blue_on()

    def door_opened(self):
        """开门：亮绿灯（保持蓝灯）"""
        self.green_on()

    def door_closed(self):
        """关门：熄灭所有灯"""
        self.all_off()

    def get_status(self) -> dict:
        """获取 LED 状态"""
        return {
            "green": self._green_on,
            "blue": self._blue_on
        }


def create_led_controller(green_pin: int, blue_pin: int) -> LEDController:
    """
    创建 LED 控制器

    Args:
        green_pin: 绿灯引脚
        blue_pin: 蓝灯引脚

    Returns:
        LED 控制器实例
    """
    return LEDController(green_pin, blue_pin)
