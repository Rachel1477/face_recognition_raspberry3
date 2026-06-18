from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session
from typing import Optional
import os
import uuid
import cv2
import numpy as np
from datetime import datetime

from app.database import get_db
from app.models import User, AccessLog
from app.schemas import LogRequest, LogResponse, AccessLogListResponse, AccessLogResponse

router = APIRouter(prefix="/logs", tags=["访问日志"])


@router.post("/log", response_model=LogResponse, summary="记录访问日志")
async def create_access_log(
    status: str = Form(..., description="访问状态：成功/失败"),
    user_id: Optional[int] = Form(None, description="用户ID（识别成功时提供）"),
    confidence: Optional[str] = Form(None, description="识别置信度"),
    image: UploadFile = File(None, description="抓拍图片"),
    db: Session = Depends(get_db)
):
    """
    记录门禁访问日志

    - **status**: 访问状态（成功/失败）
    - **user_id**: 用户ID（识别成功时必须提供）
    - **confidence**: 识别置信度
    - **image**: 抓拍图片
    """
    try:
        # 验证状态值
        if status not in ["成功", "失败"]:
            raise HTTPException(status_code=400, detail="状态必须是'成功'或'失败'")

        # 如果状态为成功，必须提供用户ID
        if status == "成功" and user_id is None:
            raise HTTPException(status_code=400, detail="识别成功时必须提供用户ID")

        # 验证用户是否存在
        if user_id is not None:
            user = db.query(User).filter(User.id == user_id).first()
            if user is None:
                raise HTTPException(status_code=404, detail="用户不存在")

        # 保存图片（如果提供）
        image_path = None
        if image:
            if not image.content_type.startswith("image/"):
                raise HTTPException(status_code=400, detail="请上传图片文件")

            # 读取图片
            contents = await image.read()
            nparr = np.frombuffer(contents, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is not None:
                # 保存图片到静态目录
                filename = f"{uuid.uuid4().hex}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                image_path = os.path.join("static", "logs", filename)

                # 确保目录存在
                os.makedirs(os.path.dirname(image_path), exist_ok=True)

                # 保存图片
                cv2.imwrite(image_path, img)

        # 创建访问日志记录
        access_log = AccessLog(
            user_id=user_id,
            status=status,
            image_path=image_path,
            confidence=confidence
        )
        db.add(access_log)
        db.commit()
        db.refresh(access_log)

        return LogResponse(
            message="访问日志记录成功",
            log_id=access_log.id,
            image_path=image_path or ""
        )

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"记录访问日志失败: {str(e)}")


@router.get("", response_model=AccessLogListResponse, summary="获取访问日志列表")
def get_access_logs(
    page: int = 1,
    page_size: int = 20,
    status: Optional[str] = None,
    user_id: Optional[int] = None,
    db: Session = Depends(get_db)
):
    """
    获取访问日志列表，支持分页和筛选

    - **page**: 页码（从1开始）
    - **page_size**: 每页记录数
    - **status**: 按状态筛选（成功/失败）
    - **user_id**: 按用户ID筛选
    """
    try:
        if status == "":
            status = None
        if user_id == "" or user_id == 0:
            user_id = None

        query = db.query(AccessLog)

        if status:
            if status not in ["成功", "失败"]:
                raise HTTPException(status_code=400, detail="状态必须是'成功'或'失败'")
            query = query.filter(AccessLog.status == status)

        if user_id is not None:
            query = query.filter(AccessLog.user_id == user_id)

        # 获取总数
        total = query.count()

        # 分页查询
        skip = (page - 1) * page_size
        logs = query.order_by(AccessLog.timestamp.desc()).offset(skip).limit(page_size).all()

        # 构建响应数据
        log_responses = []
        for log in logs:
            log_data = {
                "id": log.id,
                "user_id": log.user_id,
                "status": log.status,
                "image_path": log.image_path,
                "timestamp": log.timestamp,
                "confidence": log.confidence,
                "user": None
            }

            # 如果有关联用户，添加用户信息
            if log.user:
                log_data["user"] = {
                    "id": log.user.id,
                    "name": log.user.name,
                    "face_vector": log.user.face_vector,
                    "face_image_path": log.user.face_image_path,
                    "created_at": log.user.created_at,
                    "updated_at": log.user.updated_at
                }

            log_responses.append(AccessLogResponse(**log_data))

        return AccessLogListResponse(
            total=total,
            page=page,
            page_size=page_size,
            logs=log_responses
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取访问日志失败: {str(e)}")


@router.get("/{log_id}", response_model=AccessLogResponse, summary="获取指定访问日志")
def get_access_log(
    log_id: int,
    db: Session = Depends(get_db)
):
    """
    根据ID获取访问日志详情

    - **log_id**: 日志ID
    """
    try:
        log = db.query(AccessLog).filter(AccessLog.id == log_id).first()
        if log is None:
            raise HTTPException(status_code=404, detail="访问日志不存在")

        # 构建响应数据
        log_data = {
            "id": log.id,
            "user_id": log.user_id,
            "status": log.status,
            "image_path": log.image_path,
            "timestamp": log.timestamp,
            "confidence": log.confidence,
            "user": None
        }

        # 如果有关联用户，添加用户信息
        if log.user:
            log_data["user"] = {
                "id": log.user.id,
                "name": log.user.name,
                "face_vector": log.user.face_vector,
                "face_image_path": log.user.face_image_path,
                "created_at": log.user.created_at,
                "updated_at": log.user.updated_at
            }

        return AccessLogResponse(**log_data)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取访问日志详情失败: {str(e)}")


@router.delete("/{log_id}", summary="删除访问日志")
def delete_access_log(
    log_id: int,
    db: Session = Depends(get_db)
):
    """
    删除指定的访问日志

    - **log_id**: 日志ID
    """
    try:
        log = db.query(AccessLog).filter(AccessLog.id == log_id).first()
        if log is None:
            raise HTTPException(status_code=404, detail="访问日志不存在")

        # 删除关联的图片文件
        if log.image_path and os.path.exists(log.image_path):
            try:
                os.remove(log.image_path)
            except Exception as e:
                print(f"删除图片文件失败: {e}")

        # 删除日志记录
        db.delete(log)
        db.commit()

        return {"message": "访问日志删除成功"}

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"删除访问日志失败: {str(e)}")


@router.get("/statistics/summary", summary="获取访问统计摘要")
def get_access_statistics(
    db: Session = Depends(get_db)
):
    """
    获取访问统计摘要信息

    返回总访问次数、成功次数、失败次数等统计信息
    """
    try:
        # 总访问次数
        total_count = db.query(AccessLog).count()

        # 成功次数
        success_count = db.query(AccessLog).filter(AccessLog.status == "成功").count()

        # 失败次数
        fail_count = db.query(AccessLog).filter(AccessLog.status == "失败").count()

        # 今日访问次数
        from datetime import date
        today = date.today()
        today_count = db.query(AccessLog).filter(
            AccessLog.timestamp >= today
        ).count()

        return {
            "total_count": total_count,
            "success_count": success_count,
            "fail_count": fail_count,
            "today_count": today_count,
            "success_rate": round(success_count / total_count * 100, 2) if total_count > 0 else 0
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取访问统计失败: {str(e)}")