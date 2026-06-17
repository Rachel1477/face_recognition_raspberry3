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


class AccessLogCreate(AccessLogBase):
    """创建访问日志模型"""
    image_path: Optional[str] = None


class AccessLogResponse(AccessLogBase):
    """访问日志响应模型"""
    id: int
    image_path: Optional[str] = None
    timestamp: datetime
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