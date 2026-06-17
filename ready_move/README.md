# 树莓派人脸识别门禁系统

基于树莓派 3 的边缘人脸识别门禁系统。使用 OpenCV 官方人脸识别接口，适配 Python 3.13。

## 功能特性

- **人脸检测**：使用 YuNet（cv2.FaceDetectorYN）进行实时检测
- **人脸识别**：使用 SFace（cv2.FaceRecognizerSF）提取 128 维特征向量
- **人脸对齐**：基于 5 点关键点的仿射变换对齐（alignCrop）
- **本地比对**：余弦相似度比对（阈值 0.36，SFace 推荐值）
- **门禁控制**：通过 GPIO 控制继电器实现开门
- **远程控制**：通过 MQTT 接收后端的开门指令
- **日志上传**：自动上传访问日志到后端服务器
- **多线程架构**：充分利用树莓派 3 的四核 CPU

## 文件结构

```
ready_move/
├── config.py                    # 配置文件
├── face_embedding.py            # OpenCV Face 模块封装
├── face_recognition_system.py   # 主程序（多线程）
├── gpio_controller.py           # GPIO 控制模块
├── mqtt_client.py               # MQTT 客户端模块
├── test_local.py                # 本地测试脚本
├── requirements.txt             # Python 依赖
└── README.md                    # 说明文档
```

## 安装步骤

### 1. 安装系统依赖

```bash
sudo apt-get update
sudo apt-get install -y python3-dev python3-pip
sudo apt-get install -y libopencv-dev python3-opencv
```

### 2. 创建虚拟环境

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. 安装 Python 依赖

```bash
pip install -r requirements.txt
pip install RPi.GPIO  # 在树莓派上安装
```

### 4. 下载 ONNX 模型文件

#### YuNet 人脸检测模型
```bash
wget https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
```

#### SFace 人脸识别模型
```bash
wget https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx
```

确保模型文件放在项目根目录下：
- `face_detection_yunet_2023mar.onnx`
- `face_recognition_sface_2021dec.onnx`

### 5. 配置环境变量

```bash
export BACKEND_URL="http://your-server-ip:8000"
export MQTT_BROKER="your-mqtt-broker-ip"
export MQTT_PORT="1883"
```

## 本地测试

在没有树莓派硬件的情况下，可以使用测试脚本模拟运行：

### 1. 准备测试图片

在 `test_images` 文件夹中放入包含人脸的测试图片（jpg、png、bmp 格式）。

### 2. 运行测试

```bash
python test_local.py
```

### 3. 测试参数

| 参数 | 说明 |
|------|------|
| `-f, --folder` | 指定测试图片文件夹 |
| `-r, --register` | 从第一张图片注册测试用户 |
| `-n, --name` | 注册用户名 |
| `-l, --loop` | 循环播放图片 |
| `--no-gui` | 无 GUI 模式（用于无显示环境） |
| `--save-cache` | 保存用户缓存到文件 |
| `--load-cache` | 从文件加载用户缓存 |

### 4. 测试操作

- `q`: 退出
- `r`: 从当前帧注册用户
- `s`: 保存当前人脸到缓存

### 5. 示例

```bash
# 基本测试
python test_local.py

# 指定图片文件夹并注册用户
python test_local.py -f ./my_test_images -r -n "ZhangSan"

# 无 GUI 模式
python test_local.py -f ./test_images --no-gui

# 保存用户缓存
python test_local.py -f ./test_images --save-cache users.json
```

## 硬件连接

### 继电器接线

| 继电器引脚 | 树莓派引脚 | 说明 |
|------------|------------|------|
| VCC | 3.3V (Pin 1 或 Pin 17) | 电源 |
| GND | GND (Pin 6, 9, 14, 20, 25, 30, 34, 39) | 地线 |
| IN | GPIO 18 (Pin 12) | 控制信号 |

## 运行

```bash
python face_recognition_system.py
```

按 `q` 键退出程序。

## 配置说明

主要配置项在 `config.py` 中：

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| BACKEND_URL | 后端服务器地址 | http://localhost:8000 |
| CAMERA_INDEX | 摄像头索引 | 0 |
| FRAME_WIDTH / FRAME_HEIGHT | 摄像头分辨率 | 640 / 480 |
| FPS | 摄像头帧率 | 15 |
| FRAME_SKIP | 隔帧处理 | 3 |
| FACE_DETECTOR_MODEL | YuNet 模型路径 | face_detection_yunet_2023mar.onnx |
| FACE_RECOGNIZER_MODEL | SFace 模型路径 | face_recognition_sface_2021dec.onnx |
| COSINE_THRESHOLD | 识别相似度阈值 | 0.36 |
| DOOR_RELAY_PIN | 继电器 GPIO 引脚 | 18 |
| DOOR_UNLOCK_DURATION | 开门持续时间（秒） | 3 |
| MQTT_BROKER | MQTT 代理地址 | localhost |
| MQTT_PORT | MQTT 代理端口 | 1883 |
| USER_SYNC_INTERVAL | 用户同步间隔（秒） | 300 |

## 技术细节

### OpenCV Face 模块

本系统使用 OpenCV 4.7.0+ 引入的 Face 模块：

```python
import cv2

# 人脸检测
detector = cv2.FaceDetectorYN.create(model, "", (320, 320))
success, faces = detector.detect(image)

# 人脸识别
recognizer = cv2.FaceRecognizerSF.create(model, "")
aligned_face = recognizer.alignCrop(image, landmarks)
feature = recognizer.feature(aligned_face)
```

### 识别流程

1. **视频流读取**：640x480 分辨率，每隔 3 帧处理一次
2. **人脸检测**：YuNet 返回边界框 + 5 个关键点
3. **人脸对齐**：基于关键点的仿射变换，将人脸裁剪为 112x112
4. **特征提取**：SFace 提取 128 维特征向量
5. **余弦相似度**：与本地缓存的已知人脸特征比对
6. **阈值判断**：得分 > 0.36 判定为同一人

### 多线程架构

```
┌─────────────────────────────────────────────┐
│         主线程（视频显示）                     │
└─────────────────────────────────────────────┘
                    │
    ┌───────────────┼───────────────┐
    ▼               ▼               ▼
CaptureThread  RecognitionThread  UploadThread
(视频捕获)      (检测+识别)        (日志上传)
                    │
                    ▼
              MatchThread
              (用户匹配)
```

## 故障排除

### OpenCV 版本问题

```bash
# 检查 OpenCV 版本
python3 -c "import cv2; print(cv2.__version__)"

# 需要 4.7.0+ 以支持 Face 模块
pip install --upgrade opencv-python
```

### 模型文件问题

确保模型文件存在且格式正确：
- YuNet: `face_detection_yunet_2023mar.onnx`
- SFace: `face_recognition_sface_2021dec.onnx`

### 摄像头问题

```bash
# 检查摄像头设备
ls /dev/video*

# 添加权限
sudo usermod -a -G video $USER
```

## 环境要求

- Python 3.8+（推荐 Python 3.10+）
- OpenCV 4.7.0+
- NumPy 1.24+
- 树莓派 3 B+（四核 ARM Cortex-A53）
