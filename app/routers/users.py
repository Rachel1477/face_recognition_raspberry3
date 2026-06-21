from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session
from typing import List
import os
import uuid
import cv2
import numpy as np
import json
import logging
from datetime import datetime

from app.database import get_db
from app.models import User
from app.schemas import UserResponse, UserListResponse, RegisterResponse, ErrorResponse
from app.utils.face_utils import get_face_processor
from app.utils.speaker_utils import speaker_verifier
from app.config import VOICE_DIR

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/users", tags=["用户管理"])


@router.post("/register-complete", summary="完整注册用户（人脸+声纹）")
async def register_user_complete(
    name: str = Form(..., description="用户姓名"),
    image: UploadFile = File(..., description="人脸图片"),
    audio: UploadFile = File(..., description="声纹音频文件"),
    db: Session = Depends(get_db)
):
    """
    完整注册用户（人脸+声纹双重验证）
    
    数据完整性校验：
    - 必须同时提供人脸图片和声纹音频
    - 缺少任何一个数据都会被拒绝
    
    流程：
    1. 验证数据完整性
    2. 保存人脸图片并提取特征
    3. 保存声纹音频并提取特征
    4. 将用户信息和特征存入数据库
    
    - **name**: 用户姓名
    - **image**: 人脸图片文件
    - **audio**: 声纹音频文件（建议录制8-10秒的语音）
    """
    try:
        # ========== 数据完整性校验 ==========
        # 校验人脸图片
        if not image or not image.content_type.startswith("image/"):
            logger.warning("注册失败：缺少人脸图片或文件类型错误")
            raise HTTPException(
                status_code=400, 
                detail="数据不完整：请提供人脸图片，人脸和声纹必须同时提供"
            )
        
        # 校验声纹音频
        if not audio:
            logger.warning("注册失败：缺少声纹音频")
            raise HTTPException(
                status_code=400, 
                detail="数据不完整：请提供声纹音频，人脸和声纹必须同时提供"
            )
        
        # ========== 处理人脸图片 ==========
        # 读取图片
        image_contents = await image.read()
        nparr = np.frombuffer(image_contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            logger.warning("注册失败：无法解析人脸图片")
            raise HTTPException(status_code=400, detail="无法解析人脸图片")

        # 保存人脸图片
        face_filename = f"{uuid.uuid4().hex}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        face_image_path = os.path.join("static", "faces", face_filename)
        os.makedirs(os.path.dirname(face_image_path), exist_ok=True)
        cv2.imwrite(face_image_path, img)

        # 提取人脸特征
        face_processor = get_face_processor()
        face_vector = face_processor.extract_face_features(img)

        if face_vector is None:
            logger.warning("注册失败：未检测到人脸")
            raise HTTPException(status_code=400, detail="无法提取人脸特征：未检测到人脸")

        face_vector_json = json.dumps(face_vector.tolist())

        # ========== 处理声纹音频 ==========
        # 读取音频
        audio_bytes = await audio.read()

        # 提取声纹特征
        voice_embedding = speaker_verifier.get_embedding_from_bytes(audio_bytes)
        
        if voice_embedding is None:
            logger.warning("注册失败：无法提取声纹特征")
            raise HTTPException(
                status_code=400, 
                detail="无法提取声纹特征，请确保录制了清晰的语音（建议8-10秒）"
            )

        # 保存声纹音频
        voice_filename = f"{uuid.uuid4().hex}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        voice_audio_path = os.path.join(VOICE_DIR, voice_filename)
        os.makedirs(VOICE_DIR, exist_ok=True)
        with open(voice_audio_path, "wb") as f:
            f.write(audio_bytes)

        voice_vector_json = json.dumps(voice_embedding.tolist())

        # ========== 创建用户记录 ==========
        user = User(
            name=name,
            face_vector=face_vector_json,
            face_image_path=face_image_path,
            voice_vector=voice_vector_json,
            voice_audio_path=voice_audio_path
        )
        db.add(user)
        db.commit()
        db.refresh(user)

        logger.info(f"用户完整注册成功: id={user.id}, name={name}, 人脸+声纹已录入")

        return {
            "message": "用户注册成功（人脸+声纹）",
            "user_id": user.id,
            "face_image_path": face_image_path,
            "voice_audio_path": voice_audio_path
        }

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"完整注册失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"注册失败: {str(e)}")


@router.post("/register", response_model=RegisterResponse, summary="注册新用户（仅人脸，已废弃）")
async def register_user(
    name: str = Form(..., description="用户姓名"),
    image: UploadFile = File(..., description="人脸图片"),
    db: Session = Depends(get_db)
):
    """
    注册新用户并提取人脸特征（PC后端直接处理）
    
    流程：
    1. 保存人脸图片
    2. PC后端直接提取人脸特征
    3. 将用户信息和特征存入数据库
    
    - **name**: 用户姓名
    - **image**: 人脸图片文件
    """
    try:
        # 验证文件类型
        if not image.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="请上传图片文件")

        # 读取图片
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="无法解析图片")

        # 保存图片到静态目录
        filename = f"{uuid.uuid4().hex}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        face_image_path = os.path.join("static", "faces", filename)

        # 确保目录存在
        os.makedirs(os.path.dirname(face_image_path), exist_ok=True)

        # 保存图片
        cv2.imwrite(face_image_path, img)

        # PC后端直接提取人脸特征
        face_processor = get_face_processor()
        face_vector = face_processor.extract_face_features(img)

        if face_vector is None:
            raise HTTPException(status_code=400, detail="无法提取特征：未检测到人脸")

        # 将特征转换为JSON格式存储
        face_vector_json = json.dumps(face_vector.tolist())

        # 创建用户记录（特征已提取）
        user = User(
            name=name,
            face_vector=face_vector_json,
            face_image_path=face_image_path
        )
        db.add(user)
        db.commit()
        db.refresh(user)

        logger.info(f"用户注册成功: id={user.id}, name={name}")

        return RegisterResponse(
            message="用户注册成功",
            user_id=user.id,
            face_image_path=face_image_path
        )

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"注册失败: {str(e)}")


@router.get("", response_model=UserListResponse, summary="获取所有用户")
def get_users(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """
    获取用户列表，支持分页

    - **skip**: 跳过的记录数
    - **limit**: 返回的记录数
    """
    try:
        # 获取总数
        total = db.query(User).count()

        # 获取用户列表
        users = db.query(User).offset(skip).limit(limit).all()

        return UserListResponse(
            total=total,
            users=[UserResponse.model_validate(user) for user in users]
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取用户列表失败: {str(e)}")


@router.get("/{user_id}", response_model=UserResponse, summary="获取指定用户")
def get_user(
    user_id: int,
    db: Session = Depends(get_db)
):
    """
    根据ID获取用户信息

    - **user_id**: 用户ID
    """
    try:
        user = db.query(User).filter(User.id == user_id).first()
        if user is None:
            raise HTTPException(status_code=404, detail="用户不存在")

        return UserResponse.model_validate(user)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取用户信息失败: {str(e)}")


@router.delete("/{user_id}", summary="删除用户")
def delete_user(
    user_id: int,
    db: Session = Depends(get_db)
):
    """
    删除指定用户

    - **user_id**: 用户ID
    """
    try:
        user = db.query(User).filter(User.id == user_id).first()
        if user is None:
            raise HTTPException(status_code=404, detail="用户不存在")

        # 删除用户的人脸图片文件
        if user.face_image_path and os.path.exists(user.face_image_path):
            try:
                os.remove(user.face_image_path)
            except Exception as e:
                print(f"删除图片文件失败: {e}")

        # 删除用户的声纹音频文件
        if user.voice_audio_path and os.path.exists(user.voice_audio_path):
            try:
                os.remove(user.voice_audio_path)
            except Exception as e:
                print(f"删除音频文件失败: {e}")

        # 删除用户记录（级联删除相关的访问日志）
        db.delete(user)
        db.commit()

        return {"message": "用户删除成功"}

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"删除用户失败: {str(e)}")


@router.put("/{user_id}", response_model=UserResponse, summary="更新用户信息")
async def update_user(
    user_id: int,
    name: str = Form(..., description="用户姓名"),
    image: UploadFile = File(None, description="新的人脸图片（可选）"),
    db: Session = Depends(get_db)
):
    """
    更新用户信息

    - **user_id**: 用户ID
    - **name**: 新的用户姓名
    - **image**: 新的人脸图片（可选）
    """
    try:
        user = db.query(User).filter(User.id == user_id).first()
        if user is None:
            raise HTTPException(status_code=404, detail="用户不存在")

        # 更新姓名
        user.name = name

        # 如果上传了新图片，更新人脸特征和图片
        if image:
            if not image.content_type.startswith("image/"):
                raise HTTPException(status_code=400, detail="请上传图片文件")

            # 读取新图片
            contents = await image.read()
            nparr = np.frombuffer(contents, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is None:
                raise HTTPException(status_code=400, detail="无法解析图片")

            # 提取新的人脸特征
            face_processor = get_face_processor()
            face_vector = face_processor.extract_face_features(img)
            if face_vector is None:
                raise HTTPException(status_code=400, detail="无法检测到人脸，请上传清晰的人脸图片")

            # 删除旧图片
            if user.face_image_path and os.path.exists(user.face_image_path):
                try:
                    os.remove(user.face_image_path)
                except Exception as e:
                    print(f"删除旧图片文件失败: {e}")

            # 保存新图片
            filename = f"{uuid.uuid4().hex}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            face_image_path = os.path.join("static", "faces", filename)
            cv2.imwrite(face_image_path, img)

            # 更新用户信息
            user.face_vector = face_vector
            user.face_image_path = face_image_path

        db.commit()
        db.refresh(user)

        return UserResponse.model_validate(user)

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"更新用户失败: {str(e)}")


@router.post("/{user_id}/register-voice", summary="注册声纹")
async def register_voice(
    user_id: int,
    audio: UploadFile = File(..., description="声纹音频文件"),
    db: Session = Depends(get_db)
):
    """
    为用户注册声纹特征
    
    流程：
    1. 接收音频文件
    2. 提取声纹特征向量
    3. 保存音频文件和特征到数据库
    
    - **user_id**: 用户ID
    - **audio**: 声纹音频文件（建议录制3-5秒的语音）
    """
    try:
        user = db.query(User).filter(User.id == user_id).first()
        if user is None:
            raise HTTPException(status_code=404, detail="用户不存在")

        # 读取音频文件
        audio_bytes = await audio.read()

        # 提取声纹特征
        voice_embedding = speaker_verifier.get_embedding_from_bytes(audio_bytes)
        
        if voice_embedding is None:
            raise HTTPException(status_code=400, detail="无法提取声纹特征，请确保录制了清晰的语音")

        # 保存音频文件
        filename = f"{uuid.uuid4().hex}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        voice_audio_path = os.path.join(VOICE_DIR, filename)
        
        os.makedirs(VOICE_DIR, exist_ok=True)
        
        with open(voice_audio_path, "wb") as f:
            f.write(audio_bytes)

        # 将特征转换为JSON格式存储
        voice_vector_json = json.dumps(voice_embedding.tolist())

        # 更新用户声纹信息
        user.voice_vector = voice_vector_json
        user.voice_audio_path = voice_audio_path
        db.commit()
        db.refresh(user)

        logger.info(f"用户声纹注册成功: id={user.id}, name={user.name}")

        return {
            "message": "声纹注册成功",
            "user_id": user.id,
            "voice_audio_path": voice_audio_path
        }

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"声纹注册失败: {str(e)}")


@router.delete("/{user_id}/delete-voice", summary="删除声纹")
def delete_voice(
    user_id: int,
    db: Session = Depends(get_db)
):
    """
    删除用户的声纹信息
    
    - **user_id**: 用户ID
    """
    try:
        user = db.query(User).filter(User.id == user_id).first()
        if user is None:
            raise HTTPException(status_code=404, detail="用户不存在")

        if not user.voice_vector:
            raise HTTPException(status_code=400, detail="该用户未注册声纹")

        # 删除声纹音频文件
        if user.voice_audio_path and os.path.exists(user.voice_audio_path):
            try:
                os.remove(user.voice_audio_path)
            except Exception as e:
                print(f"删除音频文件失败: {e}")

        # 清除声纹信息
        user.voice_vector = None
        user.voice_audio_path = None
        db.commit()

        return {"message": "声纹删除成功"}

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"删除声纹失败: {str(e)}")