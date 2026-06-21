# 智能门禁系统（完整版本）

基于 **FastAPI + SQLite + SQLAlchemy + InsightFace + SpeakerVerifier + MQTT** 的智能门禁系统，支持人脸+声纹双重验证，包含后端、树莓派端和Flutter App端。

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
11. [树莓派端部署](#树莓派端部署)
12. [常见问题](#常见问题)

---

## 项目概述

智能门禁系统实现以下核心功能：

| 功能模块 | 描述 |
|---------|------|
| 人脸注册 | 接收姓名+图片，提取 512 维人脸特征向量，保存到数据库及本地磁盘 |
| 声纹注册 | 在人脸注册后，接收音频文件，提取声纹特征向量，保存到数据库 |
| 完整注册 | 同时提供人脸和声纹，确保数据完整性，缺少任一则拒绝注册 |
| 人脸对比 | 将抓拍的人脸特征与数据库中所有特征进行余弦相似度比对 |
| 声纹对比 | 将录制的语音特征与数据库中对应用户的声纹特征进行比对 |
| 双重验证 | 人脸+声纹均通过才允许开门，记录验证标签 |
| 访问日志 | 记录每次识别尝试的结果（成功/失败）、时间、抓拍图路径、匹配置信度、验证标签 |
| 人员管理 | 增删改查已注册人员，支持独立删除声纹 |
| 远程开门 | 通过 MQTT 向 `door/control` Topic 发送 OPEN 指令，记录"房主"身份 |
| 舵机控制 | 树莓派端使用 PWM 控制舵机开关门，转动到位后停止信号防抖动 |

---

## 技术栈与算法

### 后端技术栈

| 技术 | 版本 | 用途 |
|------|------|------|
| Python | 3.10+ | 运行时 |
| FastAPI | 0.110+ | Web 框架 |
| SQLAlchemy | 2.0+ | ORM |
| SQLite | 3.30+ | 数据库（轻量级） |
| InsightFace | 0.7+ | 人脸检测 + 特征提取 |
| ONNX Runtime | 1.16+ | 声纹模型推理 |
| Silero VAD | 1.0+ | 语音活动检测 |
| Librosa | 0.10+ | 音频处理 |
| OpenCV | 4.9+ | 图像预处理 |
| NumPy | 1.26+ | 数值计算 |
| Paho-MQTT | 2.0+ | MQTT 通信 |
| Uvicorn | 0.29+ | ASGI 服务器 |

### 树莓派端技术栈

| 技术 | 用途 |
|------|------|
| RPi.GPIO | GPIO 控制 |
| picamera/opencv-python | 摄像头图像采集 |
| sounddevice/arecord | 音频录制 |
| paho-mqtt | MQTT 通信 |
| luma.oled | OLED 屏幕显示 |

### Flutter App 技术栈

| 技术 | 用途 |
|------|------|
| Flutter | 跨平台移动框架 |
| flutter_sound | 录音插件（国产手机兼容） |
| permission_handler | 权限管理 |
| path_provider | 文件路径获取 |
| cached_network_image | 图片缓存 |

### 人脸识别算法

本项目采用 **InsightFace** 框架，具体使用 **buffalo_l** 模型包，包含以下子模型：

| 模型 | 作用 | 输出 |
|------|------|------|
| RetinaFace | 人脸检测 | 人脸边界框 + 5 点关键点 |
| ArcFace (iresnet100) | 特征提取 | 512 维人脸特征向量 |

**相似度计算**：采用余弦相似度（Cosine Similarity），阈值默认设为 **0.5**，大于该值判定为同一人。

### 声纹识别算法

本项目采用 **SpeakerVerifier**（基于 ONNX Runtime），包含以下核心组件：

| 组件 | 作用 | 输出 |
|------|------|------|
| Silero VAD | 语音活动检测 | 过滤静音，提取有效语音段 |
| ResNet34 (ONNX) | 声纹特征提取 | 512 维声纹特征向量 |

**相似度计算**：采用余弦相似度（Cosine Similarity），阈值默认设为 **0.6**，大于该值判定为同一人。

**声纹注册与验证流程**：
1. 注册时：录制 8-10 秒语音 → Silero VAD 提取有效语音 → ResNet34 提取特征 → 保存到数据库
2. 验证时：录制语音 → 提取特征 → 与数据库中对应用户特征比对 → 判断相似度

### 双重验证流程

```
按下按钮 → 采集图像 → 发送人脸识别请求
    │
    ▼
人脸识别结果
    │
    ├── 失败 → 记录日志（标签: 人脸匹配失败）→ 显示失败
    │
    └── 成功（status: need_voice）
        │
        ▼
    OLED 显示"请说话 (8s)" → 录音 8 秒 → 发送声纹验证请求
        │
        ▼
    声纹验证结果
        │
        ├── 通过 → 创建日志（标签: 人脸声纹验证开门）→ 开门
        │
        └── 失败 → 创建日志（标签: 人脸通过但语音失败）→ 显示失败
```

### 远程开门流程

```
手机点击"远程开门" → POST /door/remote-open
    │
    ▼
后端记录日志（标签: app主人权限一键开门，用户: 房主）
    │
    ▼
MQTT 发布 OPEN 到 door/control
    │
    ▼
树莓派接收指令 → 舵机开门 → 3秒后自动关门
```

---

## 项目结构

```
door-access-system/
│
├── app/                              # 后端代码
│   ├── __init__.py
│   ├── config.py                     # 全局配置（目录路径、阈值等）
│   ├── database.py                   # SQLAlchemy 引擎 + Session + get_db 依赖
│   │
│   ├── models/
│   │   └── __init__.py               # User 表 + AccessLog 表 ORM 定义
│   │
│   ├── schemas/
│   │   └── __init__.py               # Pydantic 模型（请求/响应校验）
│   │
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── users.py                  # 用户注册、查询、删除、声纹注册接口
│   │   ├── logs.py                   # 访问日志记录、查询接口
│   │   ├── identify.py               # 人脸识别、声纹识别接口
│   │   └── door.py                   # 远程开门接口
│   │
│   └── utils/
│       ├── face_utils.py             # 人脸检测、特征提取、比对工具
│       ├── speaker_utils.py          # 声纹提取、比对工具
│       ├── mqtt_utils.py             # MQTT 客户端封装
│       └── file_utils.py             # 文件操作工具
│
├── ready_move/                       # 树莓派端代码
│   ├── config.py                     # 树莓派配置（后端地址、GPIO、OLED等）
│   ├── face_recognition_system.py    # 主程序（人脸识别+声纹验证+开门逻辑）
│   ├── gpio_controller.py            # 舵机控制（PWM驱动）
│   ├── oled_display.py               # OLED 屏幕显示
│   ├── mqtt_client.py                # MQTT 客户端
│   └── test_local.py                 # 本地测试脚本
│
├── flutter_app/                      # Flutter App 代码
│   ├── lib/
│   │   ├── main.dart                 # App 入口
│   │   ├── models/                   # 数据模型
│   │   │   ├── user.dart             # 用户模型
│   │   │   ├── access_log.dart       # 日志模型
│   │   │   └── api_response.dart     # API 响应模型
│   │   ├── api/
│   │   │   └── api_service.dart      # API 服务封装
│   │   ├── providers/
│   │   │   └── app_provider.dart     # 状态管理
│   │   └── screens/
│   │       ├── home_screen.dart      # 首页（远程开门+统计）
│   │       ├── logs_screen.dart      # 日志页面
│   │       └── users_screen.dart     # 用户管理页面（注册+声纹录制）
│   └── pubspec.yaml                  # 依赖配置
│
├── static/                           # 静态文件存储
│   ├── faces/                        # 注册人脸原图存储目录
│   ├── access_images/                # 识别抓拍图存储目录
│   └── voices/                       # 声纹音频存储目录
│
├── main.py                           # FastAPI 应用入口（lifespan 管理）
├── init_db.py                        # 数据库初始化脚本
├── migrate_db.py                     # 数据库迁移脚本（新增字段）
├── requirements.txt                  # Python 依赖清单
├── .env                              # 环境变量（数据库、MQTT 连接信息）
└── README.md                         # 本文档
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
| voice_vector | TEXT | NULL | 声纹特征向量，JSON 数组字符串，512 维浮点数 |
| voice_audio_path | VARCHAR(255) | NULL | 声纹音频本地路径 |
| created_at | DATETIME | DEFAULT CURRENT_TIMESTAMP | 创建时间 |
| updated_at | DATETIME | DEFAULT CURRENT_TIMESTAMP ON UPDATE | 更新时间 |

### AccessLog 表（访问日志表）

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | INT | PK, AUTO_INCREMENT | 主键 |
| user_id | INT | FK -> User.id, NULLABLE | 匹配到的用户 ID，识别失败时为 NULL |
| status | VARCHAR(20) | NOT NULL | 识别结果：成功 / 失败 |
| confidence | VARCHAR(50) | NULL | 相似度得分（字符串存储） |
| image_path | VARCHAR(255) | NULL | 抓拍图本地路径 |
| verification_tag | VARCHAR(100) | NULL | 验证标签 |
| timestamp | DATETIME | DEFAULT CURRENT_TIMESTAMP | 识别时间 |

### verification_tag 标签说明

| 标签值 | 说明 |
|--------|------|
| 人脸匹配失败 | 人脸识别阶段未通过 |
| 人脸通过但语音失败 | 人脸通过，声纹验证失败 |
| 人脸声纹验证开门 | 人脸+声纹均通过，正常开门 |
| app主人权限一键开门 | 通过 App 远程开门，用户显示为"房主" |

---

## 安装与部署

### 1. 环境准备

- Python 3.10+
- SQLite（内置，无需额外安装）
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

### 4. 配置环境变量

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

### 5. 初始化数据库表

```bash
python init_db.py
```

如果是升级已有数据库，运行迁移脚本：

```bash
python migrate_db.py
```

### 6. 启动服务

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

# 人脸识别配置
COSINE_THRESHOLD = 0.5  # 人脸余弦相似度阈值

# 声纹识别配置
VOICE_THRESHOLD = 0.6  # 声纹余弦相似度阈值

# 静态文件目录
STATIC_DIR = os.path.join(BASE_DIR, "static")
FACES_DIR = os.path.join(STATIC_DIR, "faces")
ACCESS_IMAGES_DIR = os.path.join(STATIC_DIR, "access_images")
VOICE_DIR = os.path.join(STATIC_DIR, "voices")
```

> 可根据实际环境修改 `COSINE_THRESHOLD` 和 `VOICE_THRESHOLD`，值越大越严格。

### `ready_move/config.py` — 树莓派配置

```python
BACKEND_URL = "http://192.168.173.11:8000"  # 后端服务器地址
DOOR_RELAY_PIN = 18  # 舵机控制引脚（BCM编号，物理引脚Pin 12）
VOICE_RECORD_DURATION = 8  # 录音时长（秒）
```

### `.env` — 后端动态配置（必须修改！）

| 变量 | 默认值 | 必须修改 |
|------|--------|----------|
| `DB_HOST` | `localhost` | 如果数据库在远程服务器上，改为服务器 IP |
| `DB_PASSWORD` | 空 | **必须修改为你的数据库密码** |
| `MQTT_BROKER` | `localhost` | 如果使用云端 MQTT，改为云端地址 |

### 前端 `api_service.dart` 配置

前端项目中 `lib/api/api_service.dart` 定义了后端 API 的基础地址。**请务必将其中的 `localhost` 改为后端服务器的实际 IP 地址**，否则手机端无法访问。

**示例修改：**

```dart
// 错误 ❌ —— 手机端无法访问 localhost
const String BASE_URL = "http://localhost:8000";

// 正确 ✅ —— 改为后端服务器实际 IP
const String BASE_URL = "http://192.168.173.11:8000";
```

> 注意：手机和电脑必须在同一局域网内，或者后端部署到公网服务器。

---

## API 接口文档

### 用户管理

| 方法 | 路径 | 描述 | 参数 |
|------|------|------|------|
| POST | `/users/register` | 注册用户（仅人脸） | `name: str`, `image: UploadFile` |
| POST | `/users/register-complete` | 完整注册（人脸+声纹） | `name: str`, `image: UploadFile`, `audio: UploadFile` |
| POST | `/users/{user_id}/register-voice` | 单独注册声纹 | `user_id: int`, `audio: UploadFile` |
| DELETE | `/users/{user_id}/delete-voice` | 删除声纹 | `user_id: int` |
| GET | `/users` | 获取用户列表 | `skip: int=0`, `limit: int=100` |
| GET | `/users/{user_id}` | 获取用户详情 | `user_id: int` |
| DELETE | `/users/{user_id}` | 删除用户 | `user_id: int` |

### 身份识别

| 方法 | 路径 | 描述 | 参数 |
|------|------|------|------|
| POST | `/identify/` | 人脸识别（仅识别，不记录日志） | `image: UploadFile` |
| POST | `/identify/with-log` | 人脸识别并记录日志（双因子第一步） | `image: UploadFile` |
| POST | `/identify/voice-verify` | 声纹验证（双因子第二步） | `audio: UploadFile`, `user_id: int` |

### 访问日志

| 方法 | 路径 | 描述 | 参数 |
|------|------|------|------|
| POST | `/logs` | 记录日志 | `user_id: int?`, `status: str`, `image: UploadFile` |
| GET | `/logs` | 分页查询日志 | `page: int=1`, `page_size: int=20` |
| GET | `/logs/statistics/summary` | 获取统计摘要 | 无 |

### 门禁控制

| 方法 | 路径 | 描述 | 参数 |
|------|------|------|------|
| POST | `/door/remote-open` | 远程开门（MQTT） | 无 |

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

### 声纹特征提取流程

```
原始音频
   │
   ▼
Silero VAD 语音活动检测 → 提取有效语音段
   │
   ▼
ResNet34 提取 512 维声纹特征向量 → 归一化为单位向量
```

---

## 使用示例

### 1. 完整注册用户（人脸+声纹）

```bash
curl -X POST "http://localhost:8000/users/register-complete" \
  -F "name=张三" \
  -F "image=@./photo.jpg" \
  -F "audio=@./voice.wav"
```

### 2. 人脸识别（双重验证流程第一步）

```bash
curl -X POST "http://localhost:8000/identify/with-log" \
  -F "image=@./captured.jpg"
```

**返回示例（人脸通过，需要声纹验证）：**
```json
{
  "recognized": true,
  "user_id": 1,
  "name": "张三",
  "confidence": 0.85,
  "access_granted": false,
  "status": "need_voice",
  "message": "人脸验证通过，请进行声纹验证"
}
```

**返回示例（人脸失败）：**
```json
{
  "recognized": false,
  "user_id": null,
  "name": null,
  "confidence": 0.32,
  "access_granted": "人脸匹配失败",
  "status": "denied",
  "message": "人脸匹配失败"
}
```

### 3. 声纹验证（双重验证流程第二步）

```bash
curl -X POST "http://localhost:8000/identify/voice-verify?user_id=1" \
  -F "audio=@./voice_test.wav"
```

### 4. 远程开门

```bash
curl -X POST "http://localhost:8000/door/remote-open"
```

---

## 前端对接注意事项

1. **修改 `api_service.dart` 中的 `BASE_URL`**
   - 将 `localhost` 替换为后端服务器的实际 IP（如 `http://192.168.173.11:8000`）
   - 确保手机和电脑在同一局域网

2. **图片上传**
   - 人脸注册和日志记录都使用 `multipart/form-data` 格式
   - 建议图片压缩至 1MB 以内，减少传输时间

3. **录音插件**
   - 使用 `flutter_sound` 插件录制音频
   - 录制格式为 PCM16WAV（`.wav`），采样率 16000Hz，单声道
   - 需要申请麦克风权限

4. **数据完整性校验**
   - 前端在注册时检查人脸和声纹是否都已提供
   - 后端在 `register-complete` 接口中校验，缺少任一则拒绝

5. **CORS**
   - 后端已配置 `CORSMiddleware`，允许所有来源
   - 生产环境应限制 `allow_origins`

---

## 树莓派端部署

### 1. 硬件准备

| 硬件 | 说明 |
|------|------|
| 树莓派 | Raspberry Pi 4B / 5 |
| USB 摄像头 | 支持 Linux 的免驱摄像头 |
| USB 麦克风 | 支持 Linux 的免驱麦克风 |
| OLED 屏幕 | 128x64 I2C 屏幕 |
| 舵机 | SG90 或类似型号 |
| 杜邦线 | 若干 |
| 面包板 | 可选 |

### 2. 接线说明

| 舵机引脚 | 连接 |
|----------|------|
| VCC | 外部 5V 电源 |
| GND | 树莓派 GND |
| SIGNAL | GPIO 18（物理引脚 Pin 12） |

> 注意：舵机需要外部 5V 供电，不要直接从树莓派 GPIO 取电，否则可能损坏设备。

### 3. 安装依赖

```bash
sudo apt-get update
sudo apt-get install -y python3-pip python3-opencv python3-numpy \
    python3-requests python3-rpi.gpio python3-sounddevice \
    python3-scipy python3-paho-mqtt i2c-tools

pip3 install luma.oled
```

### 4. 配置音频设备

```bash
# 查看音频设备
arecord -l

# 测试录音
arecord -D plughw:2,0 -d 3 -f S16_LE -r 16000 -c 1 test.wav
```

如果 `plughw:2,0` 不是你的麦克风设备号，请修改 `gpio_controller.py` 中的设备号。

### 5. 启动门禁系统

```bash
cd ~/ready_move
python3 face_recognition_system.py
```

---

## 常见问题

**Q: 启动时报 `ModuleNotFoundError: No module named 'insightface'`？**
A: 请先执行 `pip install -r requirements.txt`。

**Q: 人脸识别精度不够？**
A: 在 `app/config.py` 中调整 `COSINE_THRESHOLD`，或确保注册照片光线充足、正脸无遮挡。

**Q: 声纹识别精度不够？**
A: 在 `app/config.py` 中调整 `VOICE_THRESHOLD`（默认 0.6），或确保注册音频环境安静、录制清晰。

**Q: 无法提取声纹特征？**
A: 请确保录制了至少 1 秒的有效语音，环境噪声不要太大，音频格式为 WAV。

**Q: 手机 App 无法连接后端？**
A: 检查 `api_service.dart` 中的 `BASE_URL` 是否为后端电脑的实际局域网 IP，而非 `localhost`。

**Q: MQTT 指令发送成功但门没开？**
A: 确保 MQTT Broker 地址正确，且门禁设备已订阅 `door/control` Topic。

**Q: 舵机一直小角度抖动？**
A: 在 `gpio_controller.py` 的 `_set_angle` 方法中，转动到位后立即 `ChangeDutyCycle(0)` 停止信号。

**Q: 树莓派录音失败 `Error querying device -1`？**
A: 使用 `arecord` 命令替代 `sounddevice` 库，在 `face_recognition_system.py` 中修改录音逻辑。

**Q: 注册时提示"数据不完整"？**
A: 确保同时提供人脸图片和声纹音频，缺少任一都无法注册。

---

## License

MIT License