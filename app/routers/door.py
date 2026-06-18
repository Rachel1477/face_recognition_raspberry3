from fastapi import APIRouter, HTTPException
from app.schemas import RemoteOpenRequest, RemoteOpenResponse
from app.utils.mqtt_utils import mqtt_client

router = APIRouter(prefix="/door", tags=["门禁控制"])


@router.post("/remote-open", response_model=RemoteOpenResponse, summary="远程开门")
async def remote_open_door(request: RemoteOpenRequest = None):
    """
    通过MQTT发送远程开门指令

    此接口会向MQTT Topic "door/control" 发送 "OPEN" 指令，
    门禁设备接收到指令后会执行开门操作。
    """
    try:
        # 检查MQTT连接状态
        if not mqtt_client.connected:
            # 尝试重新连接
            if not mqtt_client.connect():
                raise HTTPException(
                    status_code=503,
                    detail="MQTT服务不可用，无法发送开门指令"
                )

        # 发送开门指令
        success = mqtt_client.send_door_command("OPEN")

        if success:
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
        # 检查MQTT连接状态
        if not mqtt_client.connected:
            # 尝试重新连接
            if not mqtt_client.connect():
                raise HTTPException(
                    status_code=503,
                    detail="MQTT服务不可用，无法发送关门指令"
                )

        # 发送关门指令
        success = mqtt_client.send_door_command("CLOSE")

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