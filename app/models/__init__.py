from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship
from datetime import datetime
from app.database import Base

class User(Base):
    """用户表：存储用户信息和人脸特征向量"""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False, index=True)
    face_vector = Column(Text, nullable=True)  # 存储128维特征向量，JSON格式
    face_image_path = Column(String(255), nullable=True)  # 存储人脸图片路径
    voice_vector = Column(Text, nullable=True)  # 存储声纹特征向量，JSON格式
    voice_audio_path = Column(String(255), nullable=True)  # 存储声纹音频路径
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # 关联访问日志
    access_logs = relationship("AccessLog", back_populates="user", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<User(id={self.id}, name='{self.name}')>"


class AccessLog(Base):
    """访问日志表：记录门禁访问记录"""
    __tablename__ = "access_logs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)  # 允许为空，识别失败时
    status = Column(String(20), nullable=False, index=True)  # 成功/失败
    image_path = Column(String(255), nullable=True)  # 抓拍图路径
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    confidence = Column(String(50), nullable=True)  # 识别置信度
    verification_tag = Column(String(100), nullable=True)  # 验证标签：face_only / face_passed_voice_failed / face_and_voice_passed

    # 关联用户
    user = relationship("User", back_populates="access_logs")

    def __repr__(self):
        return f"<AccessLog(id={self.id}, status='{self.status}', timestamp={self.timestamp})>"