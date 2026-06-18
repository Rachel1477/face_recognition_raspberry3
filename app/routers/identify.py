"""
人脸识别接口
接收树莓派上传的实时抓拍图片，返回识别结果
"""
import os
import io
import uuid
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
    - debug: 调试信息（人脸框坐标、相似度对比详情）
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
        
        # 检测人脸位置（用于调试）
        detected_faces = face_processor.detect_faces(img)
        
        # 提取人脸特征
        query_embedding = face_processor.extract_face_features(img)
        
        # 从数据库加载所有用户特征
        users = db.query(User).filter(User.face_vector.isnot(None)).all()
        
        recognized = False
        user_id = None
        name = None
        confidence = 0.0
        status = "失败"
        similarity_details = []  # 详细相似度对比
        primary_bbox = None  # 主要人脸框
        
        # 检查是否提取到特征向量
        if query_embedding is not None and len(query_embedding) > 0 and users:
            # 获取主要人脸框（confidence 最高的）
            if detected_faces:
                primary_bbox = max(detected_faces, key=lambda x: x['confidence'])['bbox']
            
            # 与所有用户比对
            best_match = None
            best_similarity = -1.0
            
            for user in users:
                try:
                    if user.face_vector:
                        known_embedding = np.array(json.loads(user.face_vector))
                        
                        logger.info(f"正在比对: 实时特征维度={query_embedding.shape}, 库内特征维度={known_embedding.shape}, 用户={user.name}")
                        
                        similarity = face_processor.cosine_similarity(
                            query_embedding, known_embedding
                        )
                        
                        similarity_float = float(similarity)
                        similarity_details.append({
                            "user_id": user.id,
                            "name": user.name,
                            "similarity": round(similarity_float, 4),
                            "is_match": similarity_float >= COSINE_THRESHOLD
                        })
                        
                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_match = user
                            
                except Exception as e:
                    logger.error(f"比对用户 {user.id} 失败: {e}")
                    continue

            # 按相似度降序排序
            similarity_details.sort(key=lambda x: x['similarity'], reverse=True)
            
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
                if similarity_details:
                    logger.info(f"  Top 3 相似度: {[(d['name'], d['similarity']) for d in similarity_details[:3]]}")

        # 绘制人脸框（无论是否识别成功）
        debug_image_path = None
        if primary_bbox is not None:
            try:
                # 确定颜色和标签
                if recognized:
                    box_color = (0, 255, 0)  # 绿色：识别成功
                    label = f"{name} ({confidence:.2f})"
                elif detected_faces:
                    box_color = (0, 0, 255)  # 红色：未识别
                    label = "Unknown"
                else:
                    box_color = (0, 0, 255)
                    label = "Unknown"
                
                # 绘制人脸框
                img_with_box = face_processor.draw_face_box(img, primary_bbox, label, box_color)
                
                # 保存调试图片
                debug_filename = f"debug_{uuid.uuid4().hex}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                debug_image_path = os.path.join("static", "access_images", debug_filename)
                cv2.imwrite(debug_image_path, img_with_box)
            except Exception as e:
                logger.error(f"绘制人脸框失败: {e}")

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

        # 构建调试信息
        debug_info = {
            "faces_detected": len(detected_faces),
            "primary_face_bbox": primary_bbox,
            "threshold": COSINE_THRESHOLD,
            "similarity_details": similarity_details[:10],  # 只返回前10个
            "debug_image_path": debug_image_path
        }

        return {
            "recognized": recognized,
            "user_id": user_id,
            "name": name,
            "confidence": confidence,
            "access_granted": status == "成功",
            "log_id": access_log.id,
            "message": "识别成功" if recognized else f"未识别到库内人员 (最佳: {similarity_details[0]['name'] if similarity_details else 'N/A'}={confidence:.4f})",
            "debug": debug_info
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"人脸识别并记录日志失败: {e}")
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")
