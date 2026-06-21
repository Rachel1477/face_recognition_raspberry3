from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session

from app.schemas import RemoteOpenRequest, RemoteOpenResponse
from app.utils.mqtt_utils import mqtt_client
from app.database import get_db
from app.models import AccessLog

router = APIRouter(prefix="/door", tags=["门禁控制"])


@router.post("/remote-open", response_model=RemoteOpenResponse, summary="远程开门")
async def remote_open_door(request: RemoteOpenRequest = None, db: Session = Depends(get_db)):
    """
    通过MQTT发送远程开门指令

    此接口会向MQTT Topic "door/control" 发送 "OPEN" 指令，
    门禁设备接收到指令后会执行开门操作。

    成功后会记录访问日志（tag: app主人权限一键开门）
    """
    try:
        # 检查 MQTT 客户端是否存在
        if mqtt_client is None or not hasattr(mqtt_client, 'connected'):
            raise HTTPException(
                status_code=503,
                detail="MQTT服务未初始化"
            )

        # 检查MQTT连接状态
        if not getattr(mqtt_client, 'connected', False):
            # 尝试重新连接
            try:
                if not mqtt_client.connect():
                    raise HTTPException(
                        status_code=503,
                        detail="MQTT服务不可用，无法发送开门指令"
                    )
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(
                    status_code=503,
                    detail=f"MQTT服务不可用: {str(e)}"
                )

        # 发送开门指令
        try:
            success = mqtt_client.send_door_command("OPEN")
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"发送开门指令失败: {str(e)}"
            )

        if success:
            # 记录访问日志 - app主人权限一键开门
            access_log = AccessLog(
                user_id=None,
                status="成功",
                confidence=None,
                image_path=None,
                verification_tag="app主人权限一键开门"
            )
            db.add(access_log)
            db.commit()

            return RemoteOpenResponse(
                message="开门指令已发送",
                status="success"
            )
        else:
            raise HTTPException(
                status_code=500,
                detail="发送开门指令失败"
            )

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"远程开门失败: {str(e)}"
        )


@router.post("/remote-close", response_model=RemoteOpenResponse, summary="远程关门")
async def remote_close_door(request: RemoteOpenRequest = None):
    """
    通过MQTT发送远程关门指令

    此接口会向MQTT Topic "door/control" 发送 "CLOSE" 指令，
    门禁设备接收到指令后会执行关门操作。
    """
    try:
        # 检查 MQTT 客户端是否存在
        if mqtt_client is None or not hasattr(mqtt_client, 'connected'):
            raise HTTPException(
                status_code=503,
                detail="MQTT服务未初始化"
            )

        # 检查MQTT连接状态
        if not getattr(mqtt_client, 'connected', False):
            # 尝试重新连接
            try:
                if not mqtt_client.connect():
                    raise HTTPException(
                        status_code=503,
                        detail="MQTT服务不可用，无法发送关门指令"
                    )
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(
                    status_code=503,
                    detail=f"MQTT服务不可用: {str(e)}"
                )

        # 发送关门指令
        try:
            success = mqtt_client.send_door_command("CLOSE")
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"发送关门指令失败: {str(e)}"
            )

        if success:
            return RemoteOpenResponse(
                message="关门指令已发送",
                status="success"
            )
        else:
            raise HTTPException(
                status_code=500,
                detail="发送关门指令失败"
            )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"远程关门失败: {str(e)}"
        )


@router.get("/status", summary="获取门禁状态")
async def get_door_status():
    """
    获取门禁系统状态

    返回MQTT连接状态和系统运行状态
    """
    try:
        return {
            "mqtt_connected": mqtt_client.connected,
            "mqtt_broker": mqtt_client.broker,
            "mqtt_port": mqtt_client.port,
            "system_status": "running" if mqtt_client.connected else "disconnected"
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"获取门禁状态失败: {str(e)}"
        )