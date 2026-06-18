# 智能门禁系统后端（Phase 1）

基于 **FastAPI + MySQL + SQLAlchemy + InsightFace + MQTT** 的智能门禁系统后端第一阶段实现。

---

## 目录

1. [项目概述](#项目概述)
2. [技术栈与算法](#技术栈与算法)
3. [项目结构](#项目结构)
4. [数据库设计](#数据库设计)
5. [安装与部署](#安装与部署)
6. [配置文件说明](#配置文件说明)
7. [API 接口文档](#api-接口文档)
8. [算法原理](#算法原理)
9. [使用示例](#使用示例)
10. [前端对接注意事项](#前端对接注意事项)
11. [常见问题](#常见问题)

---

## 项目概述

智能门禁系统第一阶段实现以下核心功能：

| 功能模块 | 描述 |
|---------|------|
| 人脸注册 | 接收姓名+图片，提取 128 维人脸特征向量，保存到数据库及本地磁盘 |
| 人脸对比 | 将抓拍的人脸特征与数据库中所有特征进行余弦相似度比对 |
| 访问日志 | 记录每次识别尝试的结果（成功/失败）、时间、抓拍图路径、匹配置信度 |
| 人员管理 | 增删改查已注册人员 |
| 远程开门 | 通过 MQTT 向 `door/control` Topic 发送 OPEN 指令 |

---

## 技术栈与算法

### 后端技术栈

| 技术 | 版本 | 用途 |
|------|------|------|
| Python | 3.10+ | 运行时 |
| FastAPI | 0.110+ | Web 框架 |
| SQLAlchemy | 2.0+ | ORM |
| PyMySQL | 1.1+ | MySQL 驱动 |
| InsightFace | 0.7+ | 人脸检测 + 特征提取 |
| OpenCV | 4.9+ | 图像预处理 |
| NumPy | 1.26+ | 数值计算 |
| Paho-MQTT | 2.0+ | MQTT 通信 |
| Uvicorn | 0.29+ | ASGI 服务器 |

### 人脸识别算法

本项目采用 **InsightFace** 框架，具体使用 **buffalo_l** 模型包，包含以下子模型：

| 模型 | 作用 | 输出 |
|------|------|------|
| RetinaFace | 人脸检测 | 人脸边界框 + 5 点关键点 |
| ArcFace (iresnet100) | 特征提取 | 512 维人脸特征向量 |

**相似度计算**：采用余弦相似度（Cosine Similarity），阈值默认设为 **0.5**，大于该值判定为同一人。

### MQTT 通信

- **Broker**：可连接本地 Mosquitto 或阿里云/腾讯云 MQTT
- **Topic**：
  - `door/control`（发布 OPEN/CLOSE 指令）
  - `door/status`（可选，订阅门禁状态反馈）

---

## 项目结构

```
door-access-system/
│
├── app/
│   ├── __init__.py
│   ├── config.py                    # 全局配置（目录路径、阈值等）
│   ├── database.py                  # SQLAlchemy 引擎 + Session + get_db 依赖
│   │
│   ├── models/
│   │   └── __init__.py              # User 表 + AccessLog 表 ORM 定义
│   │
│   ├── schemas/
│   │   └── __init__.py              # Pydantic 模型（请求/响应校验）
│   │
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── users.py                 # 用户注册、查询、删除接口
│   │   ├── logs.py                  # 访问日志记录、查询接口
│   │   └── door.py                  # 远程开门/关门接口
│   │
│   ├── services/
│   │   └── (预留)                   # 第二阶段可放入业务逻辑层
│   │
│   └── utils/
│       ├── face_utils.py            # 人脸检测、特征提取、比对工具
│       └── mqtt_utils.py            # MQTT 客户端封装
│
├── static/
│   ├── faces/                       # 注册人脸原图存储目录
│   └── access_images/              # 识别抓拍图存储目录
│
├── models/                          # InsightFace 模型文件缓存目录
│
├── main.py                          # FastAPI 应用入口（ lifespan 管理）
├── init_db.py                       # 一键初始化数据库脚本
├── requirements.txt                 # Python 依赖清单
├── .env                             # 环境变量（数据库、MQTT 连接信息）
└── README.md                        # 本文档
```

---

## 数据库设计

### User 表（人员表）

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | INT | PK, AUTO_INCREMENT | 主键 |
| name | VARCHAR(100) | NOT NULL | 姓名 |
| face_vector | TEXT | NULL | 人脸特征向量，JSON 数组字符串，512 维浮点数 |
| face_image_path | VARCHAR(255) | NULL | 人脸原图本地路径 |
| created_at | DATETIME | DEFAULT CURRENT_TIMESTAMP | 创建时间 |
| updated_at | DATETIME | DEFAULT CURRENT_TIMESTAMP ON UPDATE | 更新时间 |

### AccessLog 表（访问日志表）

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | INT | PK, AUTO_INCREMENT | 主键 |
| user_id | INT | FK -> User.id, NULLABLE | 匹配到的用户 ID，识别失败时为 NULL |
| status | VARCHAR(20) | NOT NULL | 识别结果：success / failed |
| confidence | FLOAT | NULL | 余弦相似度得分 |
| image_path | VARCHAR(255) | NULL | 抓拍图本地路径 |
| timestamp | DATETIME | DEFAULT CURRENT_TIMESTAMP | 识别时间 |

---

## 安装与部署

### 1. 环境准备

- Python 3.10+
- MySQL 5.7+ / 8.0
- MQTT Broker（可选，本地测试可用 mosquitto）

### 2. 克隆项目

```bash
git clone <仓库地址>
cd door-access-system
```

### 3. 创建虚拟环境并安装依赖

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 4. 创建数据库

```sql
CREATE DATABASE door_access_db CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
```

### 5. 配置环境变量

在项目根目录创建 `.env` 文件，参考以下模板：

```env
# ========== 数据库配置 ==========
DB_HOST=localhost
DB_PORT=3306
DB_USER=root
DB_PASSWORD=你的数据库密码
DB_NAME=door_access_db

# ========== MQTT 配置 ==========
MQTT_BROKER=localhost
MQTT_PORT=1883
MQTT_USERNAME=（如有认证则填写）
MQTT_PASSWORD=（如有认证则填写）

# ========== 应用配置 ==========
APP_HOST=0.0.0.0
APP_PORT=8000
```

### 6. 初始化数据库表

```bash
python init_db.py
```

### 7. 启动服务

```bash
# 开发模式（热重载）
python main.py

# 或
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

启动成功后访问：
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

---

## 配置文件说明

### `app/config.py` — 后端静态配置

```python
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "app", "models")

# 余弦相似度阈值
COSINE_THRESHOLD = 0.5

# 静态文件目录
STATIC_DIR = os.path.join(BASE_DIR, "static")
FACES_DIR = os.path.join(STATIC_DIR, "faces")
ACCESS_IMAGES_DIR = os.path.join(STATIC_DIR, "access_images")
```

> 可根据实际环境修改 `COSINE_THRESHOLD`，值越大越严格。

### `.env` — 后端动态配置（必须修改！）

| 变量 | 默认值 | 必须修改 |
|------|--------|----------|
| `DB_HOST` | `localhost` | 如果数据库在远程服务器上，改为服务器 IP |
| `DB_PASSWORD` | 空 | **必须修改为你的数据库密码** |
| `MQTT_BROKER` | `localhost` | 如果使用云端 MQTT，改为云端地址 |

### 前端 `api.dart` 配置（如果你使用 Flutter 前端）

前端项目中通常有一个 `lib/services/api.dart`（或类似文件），其中定义了后端 API 的基础地址。**请务必将其中的 `localhost` 或 `127.0.0.1` 改为后端服务器的实际 IP 地址**，否则手机端无法访问。

**示例修改：**

```dart
// 错误 ❌ —— 手机端无法访问 localhost
const String BASE_URL = "http://localhost:8000";

// 正确 ✅ —— 改为后端服务器实际 IP
const String BASE_URL = "http://192.168.1.100:8000";
```

> 注意：手机和电脑必须在同一局域网内，或者后端部署到公网服务器。

---

## API 接口文档

### 用户管理

| 方法 | 路径 | 描述 | 参数 |
|------|------|------|------|
| POST | `/register` | 注册用户 | `name: str`, `image: UploadFile` |
| GET | `/users` | 获取用户列表 | `skip: int=0`, `limit: int=100` |
| DELETE | `/user/{id}` | 删除用户 | `id: int` |

### 访问日志

| 方法 | 路径 | 描述 | 参数 |
|------|------|------|------|
| POST | `/log` | 记录日志 | `user_id: int?`, `status: str`, `image: UploadFile` |
| GET | `/logs` | 分页查询日志 | `page: int=1`, `page_size: int=20` |

### 门禁控制

| 方法 | 路径 | 描述 | 参数 |
|------|------|------|------|
| POST | `/remote-open` | 远程开门 | 无（body 为空） |

---

## 算法原理

### 人脸特征提取流程

```
原始图片
   │
   ▼
RetinaFace 检测人脸 → 返回 bbox + 5 个关键点
   │
   ▼
对齐（根据关键点仿射变换）
   │
   ▼
ArcFace 提取 512 维特征向量 → 归一化为单位向量
```

### 人脸比对流程

```
抓拍图特征向量 A     数据库特征向量 B
        │                    │
        └──── 余弦相似度 ────┘
                 │
                 ▼
         score = A · B / (|A| × |B|)
                 │
         ┌──────┴──────┐
    score ≥ 0.5   score < 0.5
         │              │
      识别成功       识别失败
```

---

## 使用示例

### 1. 注册用户（curl）

```bash
curl -X POST "http://localhost:8000/register" \
  -F "name=张三" \
  -F "image=@./photo.jpg"
```

### 2. 记录识别日志

```bash
curl -X POST "http://localhost:8000/log" \
  -F "status=success" \
  -F "user_id=1" \
  -F "image=@./captured.jpg"
```

### 3. 远程开门

```bash
curl -X POST "http://localhost:8000/remote-open"
```

---

## 前端对接注意事项

1. **修改 `api.dart` 中的 `BASE_URL`**
   - 将 `localhost` 替换为后端服务器的实际 IP（如 `http://192.168.1.100:8000`）
   - 确保手机和电脑在同一局域网

2. **图片上传**
   - 人脸注册和日志记录都使用 `multipart/form-data` 格式
   - 建议图片压缩至 1MB 以内，减少传输时间

3. **CORS**
   - 后端已配置 `CORSMiddleware`，允许所有来源
   - 生产环境应限制 `allow_origins`

4. **MQTT 远程开门**
   - 手机端点击开门按钮调用 `POST /remote-open`
   - 后端通过 MQTT 发布 `OPEN` 消息到 `door/control`

---

## 常见问题

**Q: 启动时报 `ModuleNotFoundError: No module named 'insightface'`？**
A: 请先执行 `pip install -r requirements.txt`。

**Q: 人脸识别精度不够？**
A: 在 `app/config.py` 中调整 `COSINE_THRESHOLD`，或确保注册照片光线充足、正脸无遮挡。

**Q: 手机 App 无法连接后端？**
A: 检查 `api.dart` 中的 `BASE_URL` 是否为后端电脑的实际局域网 IP，而非 `localhost`。

**Q: MQTT 指令发送成功但门没开？**
A: 确保 MQTT Broker 地址正确，且门禁设备已订阅 `door/control` Topic。

---

## 下一阶段计划（Phase 2）

- [ ] 接入 Redis 做特征向量缓存，加速比对
- [ ] 接入 WebSocket 实现识别结果实时推送
- [ ] 管理员权限校验（JWT Token）
- [ ] 部署到云服务器（Docker + Nginx）

---

## License

MIT License