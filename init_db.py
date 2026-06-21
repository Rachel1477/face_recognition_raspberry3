"""
数据库初始化脚本
用于创建数据库和初始化表结构
支持数据库迁移：自动添加缺失的列
"""
import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.database import DATABASE_URL, Base, engine, SQLITE_DB_PATH
from app.models import User, AccessLog
import logging
from sqlalchemy import inspect, text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_table_exists(table_name):
    """检查表是否存在"""
    inspector = inspect(engine)
    return table_name in inspector.get_table_names()


def get_table_columns(table_name):
    """获取表的列名列表"""
    inspector = inspect(engine)
    columns = inspector.get_columns(table_name)
    return [col['name'] for col in columns]


def migrate_database():
    """数据库迁移：添加缺失的列"""
    try:
        logger.info("开始数据库迁移检查...")
        
        # 检查 users 表
        if check_table_exists('users'):
            existing_columns = get_table_columns('users')
            logger.info(f"users 表现有列: {existing_columns}")
            
            # 需要添加的列
            required_columns = {
                'voice_vector': 'TEXT',
                'voice_audio_path': 'VARCHAR(255)'
            }
            
            # 添加缺失的列
            with engine.connect() as conn:
                for col_name, col_type in required_columns.items():
                    if col_name not in existing_columns:
                        logger.info(f"添加缺失的列: users.{col_name}")
                        alter_sql = f"ALTER TABLE users ADD COLUMN {col_name} {col_type}"
                        conn.execute(text(alter_sql))
                        conn.commit()
                        logger.info(f"✓ 成功添加列: users.{col_name}")
        
        # 检查 access_logs 表
        if check_table_exists('access_logs'):
            existing_columns = get_table_columns('access_logs')
            logger.info(f"access_logs 表现有列: {existing_columns}")
            
            # 需要添加的列
            required_columns = {
                'verification_tag': 'VARCHAR(100)'
            }
            
            # 添加缺失的列
            with engine.connect() as conn:
                for col_name, col_type in required_columns.items():
                    if col_name not in existing_columns:
                        logger.info(f"添加缺失的列: access_logs.{col_name}")
                        alter_sql = f"ALTER TABLE access_logs ADD COLUMN {col_name} {col_type}"
                        conn.execute(text(alter_sql))
                        conn.commit()
                        logger.info(f"✓ 成功添加列: access_logs.{col_name}")
        
        logger.info("数据库迁移完成")
        return True
        
    except Exception as e:
        logger.error(f"数据库迁移失败: {e}")
        return False


def init_tables():
    """初始化数据库表"""
    try:
        # 创建所有表（如果不存在）
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
    logger.info("=" * 60)
    logger.info("开始初始化数据库...")
    logger.info("=" * 60)
    logger.info(f"数据库路径: {SQLITE_DB_PATH}")
    logger.info(f"数据库URL: {DATABASE_URL}")
    logger.info("")

    # 步骤1: 创建表（如果不存在）
    logger.info("步骤1: 创建数据库表...")
    if not init_tables():
        logger.error("数据库表初始化失败，终止初始化")
        sys.exit(1)
    logger.info("")

    # 步骤2: 数据库迁移（添加缺失的列）
    logger.info("步骤2: 数据库迁移（添加缺失的列）...")
    if not migrate_database():
        logger.error("数据库迁移失败，终止初始化")
        sys.exit(1)
    logger.info("")

    # 步骤3: 显示最终表结构
    logger.info("步骤3: 显示最终表结构...")
    try:
        inspector = inspect(engine)
        for table_name in ['users', 'access_logs']:
            if table_name in inspector.get_table_names():
                columns = inspector.get_columns(table_name)
                logger.info(f"\n表 {table_name} 的列:")
                for col in columns:
                    logger.info(f"  - {col['name']}: {col['type']}")
    except Exception as e:
        logger.warning(f"无法显示表结构: {e}")
    
    logger.info("")
    logger.info("=" * 60)
    logger.info("数据库初始化完成！")
    logger.info("=" * 60)
    logger.info("现在可以启动应用了: python main.py")


if __name__ == "__main__":
    main()
