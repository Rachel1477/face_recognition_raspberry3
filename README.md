# 智能门禁系统后端

基于 FastAPI + MySQL + OpenCV + MQTT 的智能门禁系统后端实现。

## 功能特性

- 人脸识别：支持人脸注册和识别
- 用户管理：添加、删除、查询用户信息
- 访问日志：记录门禁访问历史
- 远程控制：通过MQTT实现远程开门/关门
- 图片存储：自动保存人脸图片和抓拍图片

## 技术栈

- **Web框架**: FastAPI
- **数据库**: MySQL + SQLAlchemy
- **图像处理**: OpenCV + NumPy
- **消息队列**: MQTT (paho-mqtt)
- **数据验证**: Pydantic

## 项目结构

```
door-access-system/
├── app/
│   ├── models/           # 数据库模型
│   │   └── __init__.py   # User, AccessLog 模型
│   ├── schemas/          # Pydantic schemas
│   │   └── __init__.py   # 请求/响应模型
│   ├── routers/          # API路由
│   │   ├── users.py      # 用户管理接口
│   │   ├── logs.py       # 访问日志接口
│   │   └── door.py       # 门禁控制接口
│   ├── utils/            # 工具函数
│   │   ├── face_utils.py # 人脸处理工具
│   │   └── mqtt_utils.py # MQTT工具
│   └── database.py       # 数据库连接配置
├── static/               # 静态文件
│   ├── faces/           # 人脸图片
│   └── logs/            # 抓拍图片
├── main.py              # 应用入口
├── requirements.txt     # 依赖包
└── .env.example        # 环境变量示例
```

## 安装部署

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

复制 `.env.example` 为 `.env` 并修改配置：

```bash
cp .env.example .env
```

编辑 `.env` 文件，设置数据库和MQTT连接信息。

### 3. 创建数据库

```sql
CREATE DATABASE door_access_system CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
```

### 4. 启动应用

```bash
python main.py
```

或使用 uvicorn：

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## API接口文档

启动应用后，访问以下地址查看API文档：

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 主要API接口

### 用户管理

- `POST /users/register` - 注册新用户（上传人脸图片）
- `GET /users` - 获取用户列表
- `GET /users/{user_id}` - 获取指定用户信息
- `DELETE /users/{user_id}` - 删除用户
- `PUT /users/{user_id}` - 更新用户信息

### 访问日志

- `POST /logs/log` - 记录访问日志
- `GET /logs` - 获取访问日志列表（支持分页和筛选）
- `GET /logs/{log_id}` - 获取指定访问日志详情
- `DELETE /logs/{log_id}` - 删除访问日志
- `GET /logs/statistics/summary` - 获取访问统计摘要

### 门禁控制

- `POST /door/remote-open` - 远程开门
- `POST /door/remote-close` - 远程关门
- `GET /door/status` - 获取门禁状态

## 使用示例

### 注册用户

```bash
curl -X POST "http://localhost:8000/users/register" \
  -F "name=张三" \
  -F "image=@face.jpg"
```

### 远程开门

```bash
curl -X POST "http://localhost:8000/door/remote-open" \
  -H "Content-Type: application/json"
```

### 查询访问日志

```bash
curl -X GET "http://localhost:8000/logs?page=1&page_size=20"
```

## 注意事项

1. 确保MySQL服务正在运行
2. 确保MQTT Broker正在运行（如果需要远程控制功能）
3. 人脸识别功能需要清晰的正脸图片
4. 生产环境请修改 `.env` 中的默认密码
5. 建议使用HTTPS保护API接口

## 开发说明

- 人脸识别使用OpenCV的Haar级联分类器和LBPH识别器
- 特征向量存储为JSON格式
- MQTT Topic: `door/control`
- 支持CORS跨域请求

## 许可证

MIT License