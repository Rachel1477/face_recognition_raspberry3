"""
人脸识别接口
接收树莓派上传的实时抓拍图片，返回识别结果
支持人脸+声纹双重验证
"""
import os
import uuid
import cv2
import numpy as np
import json
import logging
from datetime import datetime

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Query
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import User, AccessLog
from app.utils.face_utils import get_face_processor
from app.utils.speaker_utils import speaker_verifier
from app.config import COSINE_THRESHOLD, VOICE_THRESHOLD

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/identify", tags=["身份识别"])


@router.post("/", summary="人脸识别")
async def identify_face(
    image: UploadFile = File(..., description="抓拍图片"),
    db: Session = Depends(get_db)
):
    """
    接收树莓派上传的实时抓拍图片，进行人脸识别
    
    返回结果：
    - recognized: 是否识别成功
    - user_id: 用户ID（识别成功时）
    - name: 用户姓名（识别成功时）
    - confidence: 置信度
    - message: 状态消息
    """
    try:
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="无法解析图片")

        face_processor = get_face_processor()
        query_embedding = face_processor.extract_face_features(img)
        
        if query_embedding is None:
            return {
                "recognized": False,
                "user_id": None,
                "name": None,
                "confidence": 0.0,
                "message": "未检测到人脸"
            }

        users = db.query(User).filter(User.face_vector.isnot(None)).all()
        
        if not users:
            return {
                "recognized": False,
                "user_id": None,
                "name": None,
                "confidence": 0.0,
                "message": "用户库为空"
            }

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


@router.post("/with-log", summary="人脸识别（双因子第一步）")
async def identify_and_log(
    image: UploadFile = File(..., description="抓拍图片"),
    db: Session = Depends(get_db)
):
    """
    双因子验证第一步：仅验证人脸
    
    人脸匹配成功 → 返回 need_voice，等待声纹验证
    人脸匹配失败 → 返回失败，记录日志
    
    返回结果：
    - recognized: 是否识别成功
    - user_id: 用户ID（识别成功时）
    - name: 用户姓名（识别成功时）
    - confidence: 置信度
    - status: need_voice / denied
    - access_granted: 人脸匹配失败时为"人脸匹配失败"，成功时为false
    - log_id: 日志ID（人脸失败时）
    - message: 状态消息
    """
    try:
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="无法解析图片")

        from app.utils.file_utils import save_access_image
        access_image_path = await save_access_image(img)

        face_processor = get_face_processor()
        detected_faces = face_processor.detect_faces(img)
        query_embedding = face_processor.extract_face_features(img)
        
        users = db.query(User).filter(User.face_vector.isnot(None)).all()
        
        recognized = False
        user_id = None
        name = None
        confidence = 0.0
        status = "denied"
        access_granted = "人脸匹配失败"
        log_id = None
        similarity_details = []
        primary_bbox = None
        
        if query_embedding is not None and len(query_embedding) > 0 and users:
            if detected_faces:
                primary_bbox = max(detected_faces, key=lambda x: x['confidence'])['bbox']
            
            best_match = None
            best_similarity = -1.0
            
            for user in users:
                try:
                    if user.face_vector:
                        known_embedding = np.array(json.loads(user.face_vector))
                        
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

            similarity_details.sort(key=lambda x: x['similarity'], reverse=True)
            
            if best_match and best_similarity >= COSINE_THRESHOLD:
                recognized = True
                user_id = best_match.id
                name = best_match.name
                confidence = round(float(best_similarity), 4)
                status = "need_voice"
                access_granted = False
                
                logger.info(f"人脸验证成功: {name} (置信度: {confidence})，等待声纹验证")
            else:
                logger.info(f"人脸验证失败: 最佳匹配置信度 {best_similarity:.4f}")

        # 人脸验证失败时创建访问日志
        if status == "denied":
            access_log = AccessLog(
                user_id=user_id,
                status="失败",
                confidence=str(confidence) if confidence > 0 else None,
                image_path=access_image_path,
                verification_tag="人脸匹配失败"
            )
            db.add(access_log)
            db.commit()
            db.refresh(access_log)
            
            log_id = access_log.id

        # 绘制人脸框
        debug_image_path = None
        if primary_bbox is not None:
            try:
                if recognized:
                    box_color = (0, 255, 0)
                    label = f"{name} ({confidence:.2f})"
                elif detected_faces:
                    box_color = (0, 0, 255)
                    label = "Unknown"
                else:
                    box_color = (0, 0, 255)
                    label = "Unknown"
                
                img_with_box = face_processor.draw_face_box(img, primary_bbox, label, box_color)
                
                debug_filename = f"debug_{uuid.uuid4().hex}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                debug_image_path = os.path.join("static", "access_images", debug_filename)
                cv2.imwrite(debug_image_path, img_with_box)
            except Exception as e:
                logger.error(f"绘制人脸框失败: {e}")

        debug_info = {
            "faces_detected": len(detected_faces),
            "primary_face_bbox": primary_bbox,
            "threshold": COSINE_THRESHOLD,
            "similarity_details": similarity_details[:10],
            "debug_image_path": debug_image_path
        }

        return {
            "recognized": recognized,
            "user_id": user_id,
            "name": name,
            "confidence": confidence,
            "access_granted": access_granted,
            "status": status,
            "log_id": log_id,
            "message": "人脸验证通过，请进行声纹验证" if recognized else f"人脸匹配失败 (最佳: {similarity_details[0]['name'] if similarity_details else 'N/A'}={confidence:.4f})",
            "debug": debug_info
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"人脸识别失败: {e}")
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")


@router.post("/voice-verify", summary="声纹验证（双因子第二步）")
async def voice_verify(
    audio: UploadFile = File(..., description="语音文件"),
    user_id: int = Query(..., description="用户ID"),
    db: Session = Depends(get_db)
):
    """
    双因子验证第二步：声纹验证
    
    接收 user_id 和音频文件，进行声纹比对：
    - 声纹通过 → 创建成功日志，返回 access_granted: true
    - 声纹失败 → 创建失败日志，返回 access_granted: false
    
    返回结果：
    - access_granted: 是否允许通行
    - user_id: 用户ID
    - confidence: 声纹相似度
    - verification_tag: 验证标签
    - log_id: 日志ID
    - message: 验证结果消息
    """
    try:
        # 获取用户信息
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="用户不存在")

        if not user.voice_vector:
            raise HTTPException(status_code=400, detail="该用户未注册声纹")

        # 读取音频文件
        audio_bytes = await audio.read()

        # 提取上传语音的声纹特征
        test_embedding = speaker_verifier.get_embedding_from_bytes(audio_bytes)
        
        if test_embedding is None:
            logger.error("无法提取上传语音的声纹特征")
            
            access_log = AccessLog(
                user_id=user_id,
                status="失败",
                confidence="0",
                verification_tag="人脸通过但语音失败"
            )
            db.add(access_log)
            db.commit()
            db.refresh(access_log)
            
            return {
                "access_granted": False,
                "user_id": user_id,
                "confidence": 0.0,
                "verification_tag": "人脸通过但语音失败",
                "log_id": access_log.id,
                "message": "无法提取语音特征，请重试"
            }

        # 获取用户注册的声纹特征
        reference_embedding = np.array(json.loads(user.voice_vector))

        # 计算余弦相似度
        similarity = speaker_verifier.cosine_similarity(test_embedding, reference_embedding)
        confidence = round(float(similarity), 4)

        # 判断是否通过
        if similarity >= VOICE_THRESHOLD:
            access_granted = True
            status = "成功"
            verification_tag = "人脸声纹验证开门"
            logger.info(f"声纹验证成功: {user.name} (相似度: {confidence})")
        else:
            access_granted = False
            status = "失败"
            verification_tag = "人脸通过但语音失败"
            logger.info(f"声纹验证失败: {user.name} (相似度: {confidence} < 阈值 {VOICE_THRESHOLD})")

        # 创建访问日志
        access_log = AccessLog(
            user_id=user_id,
            status=status,
            confidence=str(confidence),
            verification_tag=verification_tag
        )
        db.add(access_log)
        db.commit()
        db.refresh(access_log)

        return {
            "access_granted": access_granted,
            "user_id": user_id,
            "confidence": confidence,
            "verification_tag": verification_tag,
            "log_id": access_log.id,
            "message": "声纹验证通过，允许通行" if access_granted else f"声纹验证失败 (相似度: {confidence})"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"声纹验证失败: {e}")
        raise HTTPException(status_code=500, detail=f"声纹验证失败: {str(e)}")