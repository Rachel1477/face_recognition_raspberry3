"""
人脸识别接口
接收树莓派上传的实时抓拍图片，返回识别结果
"""
import os
import io
import cv2
import numpy as np
import json
import logging
from typing import Optional
from datetime import datetime

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import User, AccessLog
from app.utils.face_utils import get_face_processor
from app.config import COSINE_THRESHOLD

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/identify", tags=["人脸识别"])


@router.post("/", summary="人脸识别")
async def identify_face(
    image: UploadFile = File(..., description="抓拍图片"),
    db: Session = Depends(get_db)
):
    """
    接收树莓派上传的实时抓拍图片，进行人脸识别
    
    流程：
    1. 接收图片
    2. 提取人脸特征
    3. 与数据库中的用户特征比对
    4. 返回识别结果
    
    返回结果：
    - recognized: 是否识别成功
    - user_id: 用户ID（识别成功时）
    - name: 用户姓名（识别成功时）
    - confidence: 置信度
    - message: 状态消息
    """
    try:
        # 读取图片
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="无法解析图片")

        # 获取人脸处理器
        face_processor = get_face_processor()
        
        # 提取人脸特征
        query_embedding = face_processor.extract_face_features(img)
        
        if query_embedding is None:
            return {
                "recognized": False,
                "user_id": None,
                "name": None,
                "confidence": 0.0,
                "message": "未检测到人脸"
            }

        # 从数据库加载所有用户特征
        users = db.query(User).filter(User.face_vector.isnot(None)).all()
        
        if not users:
            return {
                "recognized": False,
                "user_id": None,
                "name": None,
                "confidence": 0.0,
                "message": "用户库为空"
            }

        # 与所有用户比对
        best_match = None
        best_similarity = -1.0
        
        for user in users:
            try:
                if user.face_vector:
                    known_embedding = np.array(json.loads(user.face_vector))
                    similarity = face_processor.cosine_similarity(
                        query_embedding, known_embedding
                    )
                    
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match = user
                        
            except Exception as e:
                logger.error(f"比对用户 {user.id} 失败: {e}")
                continue

        # 判断是否识别成功
        if best_match and best_similarity >= COSINE_THRESHOLD:
            logger.info(f"识别成功: {best_match.name} (置信度: {best_similarity:.4f})")
            
            return {
                "recognized": True,
                "user_id": best_match.id,
                "name": best_match.name,
                "confidence": round(float(best_similarity), 4),
                "message": "识别成功"
            }
        else:
            logger.info(f"识别失败: 最佳匹配置信度 {best_similarity:.4f} < 阈值 {COSINE_THRESHOLD}")
            
            return {
                "recognized": False,
                "user_id": None,
                "name": None,
                "confidence": round(float(best_similarity), 4) if best_similarity > 0 else 0.0,
                "message": f"未识别到库内人员 (置信度: {best_similarity:.4f})"
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"人脸识别失败: {e}")
        raise HTTPException(status_code=500, detail=f"识别失败: {str(e)}")


@router.post("/with-log", summary="人脸识别并记录日志")
async def identify_and_log(
    image: UploadFile = File(..., description="抓拍图片"),
    db: Session = Depends(get_db)
):
    """
    接收树莓派上传的实时抓拍图片，进行人脸识别，并自动记录访问日志
    
    返回结果：
    - recognized: 是否识别成功
    - user_id: 用户ID（识别成功时）
    - name: 用户姓名（识别成功时）
    - confidence: 置信度
    - access_granted: 是否允许通行
    - log_id: 日志ID
    """
    try:
        # 读取图片
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="无法解析图片")

        # 保存抓拍图片
        from app.utils.file_utils import save_access_image
        access_image_path = await save_access_image(img)

        # 获取人脸处理器
        face_processor = get_face_processor()
        
        # 提取人脸特征
        query_embedding = face_processor.extract_face_features(img)
        
        # 从数据库加载所有用户特征
        users = db.query(User).filter(User.face_vector.isnot(None)).all()
        
        recognized = False
        user_id = None
        name = None
        confidence = 0.0
        status = "失败"
        
        if query_embedding and users:
            # 与所有用户比对
            best_match = None
            best_similarity = -1.0
            
            for user in users:
                try:
                    if user.face_vector:
                        known_embedding = np.array(json.loads(user.face_vector))
                        similarity = face_processor.cosine_similarity(
                            query_embedding, known_embedding
                        )
                        
                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_match = user
                            
                except Exception as e:
                    logger.error(f"比对用户 {user.id} 失败: {e}")
                    continue

            # 判断是否识别成功
            if best_match and best_similarity >= COSINE_THRESHOLD:
                recognized = True
                user_id = best_match.id
                name = best_match.name
                confidence = round(float(best_similarity), 4)
                status = "成功"
                
                logger.info(f"识别成功: {name} (置信度: {confidence})")
            else:
                logger.info(f"识别失败: 最佳匹配置信度 {best_similarity:.4f}")

        # 创建访问日志
        access_log = AccessLog(
            user_id=user_id,
            status=status,
            confidence=str(confidence) if confidence > 0 else None,
            image_path=access_image_path
        )
        db.add(access_log)
        db.commit()
        db.refresh(access_log)

        return {
            "recognized": recognized,
            "user_id": user_id,
            "name": name,
            "confidence": confidence,
            "access_granted": status == "成功",
            "log_id": access_log.id,
            "message": "识别成功" if recognized else "未识别到库内人员"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"人脸识别并记录日志失败: {e}")
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")
