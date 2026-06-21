"""
数据库迁移脚本
使用 sqlite3 标准库添加缺失的列
"""
import sqlite3
import os

# 数据库文件路径
DB_PATH = "door_access_system.db"

def migrate_database():
    """数据库迁移：添加缺失的列"""
    if not os.path.exists(DB_PATH):
        print(f"错误：数据库文件 {DB_PATH} 不存在")
        print("请先启动后端服务创建数据库，或检查数据库路径")
        return False
    
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        print("=" * 60)
        print("开始数据库迁移...")
        print("=" * 60)
        
        # 获取 users 表的列信息
        cursor.execute("PRAGMA table_info(users)")
        users_columns = [col[1] for col in cursor.fetchall()]
        print(f"\nusers 表现有列: {users_columns}")
        
        # 添加缺失的列到 users 表
        if 'voice_vector' not in users_columns:
            print("添加列: users.voice_vector")
            cursor.execute("ALTER TABLE users ADD COLUMN voice_vector TEXT")
            print("[OK] 成功添加列: users.voice_vector")
        
        if 'voice_audio_path' not in users_columns:
            print("添加列: users.voice_audio_path")
            cursor.execute("ALTER TABLE users ADD COLUMN voice_audio_path VARCHAR(255)")
            print("[OK] 成功添加列: users.voice_audio_path")
        
        # 获取 access_logs 表的列信息
        cursor.execute("PRAGMA table_info(access_logs)")
        logs_columns = [col[1] for col in cursor.fetchall()]
        print(f"\naccess_logs 表现有列: {logs_columns}")
        
        # 添加缺失的列到 access_logs 表
        if 'verification_tag' not in logs_columns:
            print("添加列: access_logs.verification_tag")
            cursor.execute("ALTER TABLE access_logs ADD COLUMN verification_tag VARCHAR(100)")
            print("[OK] 成功添加列: access_logs.verification_tag")
        
        conn.commit()
        
        # 显示最终表结构
        print("\n" + "=" * 60)
        print("最终表结构:")
        print("=" * 60)
        
        cursor.execute("PRAGMA table_info(users)")
        print("\nusers 表:")
        for col in cursor.fetchall():
            print(f"  - {col[1]}: {col[2]}")
        
        cursor.execute("PRAGMA table_info(access_logs)")
        print("\naccess_logs 表:")
        for col in cursor.fetchall():
            print(f"  - {col[1]}: {col[2]}")
        
        conn.close()
        
        print("\n" + "=" * 60)
        print("数据库迁移完成！")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"\n错误：数据库迁移失败: {e}")
        return False


if __name__ == "__main__":
    migrate_database()
