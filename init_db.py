"""
数据库初始化脚本
用于创建数据库和初始化表结构
"""
import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.database import DATABASE_URL, Base, engine
from app.models import User, AccessLog
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def init_tables():
    """初始化数据库表"""
    try:
        # 创建所有表
        Base.metadata.create_all(bind=engine)
        logger.info("数据库表初始化成功")

        # 显示创建的表
        logger.info("已创建的表:")
        for table_name in Base.metadata.tables.keys():
            logger.info(f"  - {table_name}")

        return True

    except Exception as e:
        logger.error(f"初始化数据库表失败: {e}")
        return False


def main():
    """主函数"""
    logger.info("开始初始化数据库...")
    logger.info(f"数据库路径: {DATABASE_URL}")

    # 初始化表
    if not init_tables():
        logger.error("数据库表初始化失败，终止初始化")
        sys.exit(1)

    logger.info("数据库初始化完成！")
    logger.info("现在可以启动应用了: python main.py")


if __name__ == "__main__":
    main()