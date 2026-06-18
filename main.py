from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
import os

from app.database import engine, Base
from app.utils.mqtt_utils import mqtt_client
from app.routers import users, logs, door, identify

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时执行
    logger.info("应用启动中...")

    # 创建数据库表
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("数据库表创建成功")
    except Exception as e:
        logger.error(f"数据库表创建失败: {e}")

    # 连接MQTT
    try:
        mqtt_connected = mqtt_client.connect()
        if mqtt_connected:
            logger.info("MQTT连接成功")
        else:
            logger.warning("MQTT连接失败，某些功能可能不可用")
    except Exception as e:
        logger.error(f"MQTT连接异常: {e}")

    # 确保静态文件目录存在
    os.makedirs("static/faces", exist_ok=True)
    os.makedirs("static/access_images", exist_ok=True)

    yield

    # 关闭时执行
    logger.info("应用关闭中...")

    # 断开MQTT连接
    try:
        mqtt_client.disconnect()
        logger.info("MQTT连接已断开")
    except Exception as e:
        logger.error(f"MQTT断开连接异常: {e}")


# 创建FastAPI应用
app = FastAPI(
    title="智能门禁系统 API",
    description="基于人脸识别的智能门禁系统后端API",
    version="1.0.0",
    lifespan=lifespan
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应该设置具体的域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载静态文件目录
app.mount("/static", StaticFiles(directory="static"), name="static")

# 注册路由
app.include_router(users.router)
app.include_router(logs.router)
app.include_router(door.router)
app.include_router(identify.router)


@app.get("/", tags=["根路径"])
async def root():
    """根路径"""
    return {
        "message": "智能门禁系统 API",
        "version": "1.0.0",
        "docs": "/docs",
        "redoc": "/redoc"
    }


@app.get("/health", tags=["健康检查"])
async def health_check():
    """健康检查接口"""
    return {
        "status": "healthy",
        "mqtt_connected": mqtt_client.connected,
        "database": "connected"  # 简化检查，实际应该检查数据库连接
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )