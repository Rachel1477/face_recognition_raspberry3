"""
OLED 屏幕控制器 - SSD1306
使用 I2C 接口，支持中文显示
"""
import logging
from typing import Optional

import board
import busio
import adafruit_ssd1306
from PIL import Image, ImageDraw, ImageFont

from config import OLED_WIDTH, OLED_HEIGHT, OLED_I2C_ADDR, OLED_FONT_PATH, OLED_FONT_SIZE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OLEDDisplay:
    """
    SSD1306 OLED 屏幕控制器
    
    支持状态：
    - 待激活 (IDLE)
    - 采集成功 (CAPTURED)
    - 正在识别... (RECOGNIZING)
    - 人脸验证成功 (SUCCESS)
    - 人脸验证失败 (FAILED)
    """
    
    def __init__(self):
        logger.info("初始化 OLED 屏幕...")
        self.width = OLED_WIDTH
        self.height = OLED_HEIGHT
        
        try:
            # 初始化 I2C
            self.i2c = busio.I2C(board.SCL, board.SDA)
            
            # 初始化 SSD1306
            self.oled = adafruit_ssd1306.SSD1306_I2C(
                self.width,
                self.height,
                self.i2c,
                addr=OLED_I2C_ADDR
            )
            
            # 清除屏幕
            self.oled.fill(0)
            self.oled.show()
            
            # 加载字体
            try:
                self.font = ImageFont.truetype(OLED_FONT_PATH, OLED_FONT_SIZE)
            except Exception as e:
                logger.warning(f"加载自定义字体失败: {e}，使用默认字体")
                self.font = ImageFont.load_default()
            
            # 创建绘图对象
            self.image = Image.new('1', (self.width, self.height))
            self.draw = ImageDraw.Draw(self.image)
            
            # 当前状态
            self.current_status = None
            
            logger.info("OLED 屏幕初始化完成")
            
        except Exception as e:
            logger.error(f"OLED 屏幕初始化失败: {e}")
            self.oled = None
    
    def clear(self):
        """清除屏幕"""
        if self.oled:
            self.oled.fill(0)
            self.oled.show()
    
    def _draw_text_centered(self, text: str, y_offset: int = 0):
        """
        在屏幕中央绘制文本
        
        Args:
            text: 要显示的文本
            y_offset: Y 轴偏移量
        """
        if not self.oled:
            return
            
        # 清除之前的内容
        self.draw.rectangle((0, 0, self.width, self.height), outline=0, fill=0)
        
        # 计算文本尺寸和位置
        bbox = self.draw.textbbox((0, 0), text, font=self.font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        x = (self.width - text_width) // 2
        y = (self.height - text_height) // 2 + y_offset
        
        # 绘制文本
        self.draw.text((x, y), text, font=self.font, fill=255)
        
        # 更新屏幕
        self.oled.image(self.image)
        self.oled.show()
    
    def _draw_multiline_text(self, lines: list, align: str = 'center'):
        """
        在屏幕上绘制多行文本
        
        Args:
            lines: 文本行列表
            align: 对齐方式 ('center', 'left', 'right')
        """
        if not self.oled:
            return
            
        # 清除之前的内容
        self.draw.rectangle((0, 0, self.width, self.height), outline=0, fill=0)
        
        bbox = self.draw.textbbox((0, 0), '测', font=self.font)
        line_height = bbox[3] - bbox[1]
        
        total_height = len(lines) * line_height
        start_y = (self.height - total_height) // 2
        
        for i, line in enumerate(lines):
            bbox = self.draw.textbbox((0, 0), line, font=self.font)
            text_width = bbox[2] - bbox[0]
            
            if align == 'center':
                x = (self.width - text_width) // 2
            elif align == 'right':
                x = self.width - text_width - 4
            else:
                x = 4
                
            y = start_y + i * line_height
            self.draw.text((x, y), line, font=self.font, fill=255)
        
        # 更新屏幕
        self.oled.image(self.image)
        self.oled.show()
    
    def show_idle(self):
        """显示待激活状态"""
        if self.current_status == 'IDLE':
            return
        self.current_status = 'IDLE'
        self._draw_text_centered("待激活")
        logger.info("屏幕显示: 待激活")
    
    def show_captured(self):
        """显示采集成功状态"""
        if self.current_status == 'CAPTURED':
            return
        self.current_status = 'CAPTURED'
        self._draw_text_centered("采集成功")
        logger.info("屏幕显示: 采集成功")
    
    def show_recognizing(self):
        """显示正在识别状态"""
        if self.current_status == 'RECOGNIZING':
            return
        self.current_status = 'RECOGNIZING'
        self._draw_multiline_text(["采集成功", "正在识别..."])
        logger.info("屏幕显示: 正在识别...")
    
    def show_success(self, user_name: Optional[str] = None):
        """显示验证成功状态"""
        if self.current_status == 'SUCCESS':
            return
        self.current_status = 'SUCCESS'
        
        if user_name:
            self._draw_multiline_text(["人脸验证", f"成功: {user_name}"])
        else:
            self._draw_text_centered("人脸验证成功")
        
        logger.info(f"屏幕显示: 人脸验证成功 {user_name}")
    
    def show_failed(self):
        """显示验证失败状态"""
        if self.current_status == 'FAILED':
            return
        self.current_status = 'FAILED'
        self._draw_text_centered("人脸验证失败")
        logger.info("屏幕显示: 人脸验证失败")
    
    def show_text(self, text: str):
        """显示自定义文本"""
        self._draw_text_centered(text)
        logger.info(f"屏幕显示: {text}")
    
    def cleanup(self):
        """清理资源"""
        logger.info("清理 OLED 屏幕资源")
        self.clear()
        if hasattr(self, 'i2c'):
            try:
                self.i2c.deinit()
            except:
                pass