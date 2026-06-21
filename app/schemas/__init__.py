from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, List


class UserBase(BaseModel):
    """用户基础模型"""
    name: str = Field(..., description="用户姓名", min_length=1, max_length=100)


class UserCreate(UserBase):
    """创建用户模型"""
    pass


class UserResponse(UserBase):
    """用户响应模型"""
    id: int
    face_vector: Optional[str] = None
    face_image_path: Optional[str] = None
    voice_vector: Optional[str] = None
    voice_audio_path: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class UserListResponse(BaseModel):
    """用户列表响应模型"""
    total: int
    users: List[UserResponse]


class AccessLogBase(BaseModel):
    """访问日志基础模型"""
    user_id: Optional[int] = None
    status: str = Field(..., description="访问状态：成功/失败")
    confidence: Optional[str] = None
    verification_tag: Optional[str] = None


class AccessLogCreate(AccessLogBase):
    """创建访问日志模型"""
    image_path: Optional[str] = None


class AccessLogResponse(AccessLogBase):
    """访问日志响应模型"""
    id: int
    image_path: Optional[str] = None
    timestamp: datetime
    verification_tag: Optional[str] = None
    user: Optional[UserResponse] = None

    class Config:
        from_attributes = True


class AccessLogListResponse(BaseModel):
    """访问日志列表响应模型"""
    total: int
    page: int
    page_size: int
    logs: List[AccessLogResponse]


class RegisterRequest(BaseModel):
    """注册请求模型"""
    name: str = Field(..., description="用户姓名", min_length=1, max_length=100)


class RegisterResponse(BaseModel):
    """注册响应模型"""
    message: str
    user_id: int
    face_image_path: str


class LogRequest(BaseModel):
    """日志记录请求模型"""
    user_id: Optional[int] = None
    status: str = Field(..., description="访问状态：成功/失败")
    confidence: Optional[str] = None


class LogResponse(BaseModel):
    """日志记录响应模型"""
    message: str
    log_id: int
    image_path: str


class RemoteOpenRequest(BaseModel):
    """远程开门请求模型"""
    pass


class RemoteOpenResponse(BaseModel):
    """远程开门响应模型"""
    message: str
    status: str


class ErrorResponse(BaseModel):
    """错误响应模型"""
    error: str
    detail: Optional[str] = None


class FeatureUpdateRequest(BaseModel):
    """特征更新请求模型（树莓派回调）"""
    user_id: int = Field(..., description="用户ID")
    face_vector: str = Field(..., description="人脸特征向量（JSON字符串）")
    success: bool = Field(..., description="是否成功提取特征")
    error_message: Optional[str] = Field(None, description="错误信息")


class FeatureUpdateResponse(BaseModel):
    """特征更新响应模型"""
    message: str
    user_id: int


class VoiceRegisterRequest(BaseModel):
    """声纹注册请求模型"""
    user_id: int = Field(..., description="用户ID")


class VoiceRegisterResponse(BaseModel):
    """声纹注册响应模型"""
    message: str
    user_id: int
    voice_audio_path: str


class VoiceIdentifyRequest(BaseModel):
    """声纹识别请求模型"""
    user_id: int = Field(..., description="用户ID（人脸匹配成功后传入）")


class VoiceIdentifyResponse(BaseModel):
    """声纹识别响应模型"""
    success: bool = Field(..., description="是否识别成功")
    user_id: Optional[int] = None
    confidence: Optional[float] = None
    message: str = Field(..., description="识别结果消息")


class IdentifyWithLogResponse(BaseModel):
    """带日志的识别响应模型"""
    success: bool = Field(..., description="是否完全通过（人脸+声纹）")
    status: str = Field(..., description="状态：access_granted / need_voice / denied")
    user_id: Optional[int] = None
    message: str = Field(..., description="识别结果消息")