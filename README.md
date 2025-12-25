# Gst-Videorecognition

基于 NVIDIA DeepStream 的视频行为识别 GStreamer 插件，使用 X3D (SlowFast) 模型实现实时视频分类。

## 概述

本插件作为 DeepStream 流水线的一部分，对视频流中的目标区域进行时序行为分析识别。它接收目标检测/跟踪模块的 ROI 区域，通过多帧采样和 X3D 模型推理，输出行为分类结果。

## 项目结构

```
gst-videorecognition/
├── CMakeLists.txt              # CMake 构建配置
├── gstvideorecognition.h       # GStreamer 插件头文件
├── gstvideorecognition.cpp     # GStreamer 插件主实现
├── videorecognitionTrt.h       # TensorRT 基础类头文件
├── videorecognitionTrt.cpp     # TensorRT 基础类实现
├── x3dTrt.h                    # X3D 模型 TensorRT 封装
├── x3dTrt.cpp                  # X3D 模型实现
├── process.h                   # 视频帧预处理头文件
├── process.cpp                 # 视频帧预处理实现
├── logging.h                   # TensorRT 日志工具
├── models/
│   └── convert2trt.sh          # ONNX 转 TensorRT 引擎脚本
└── build/                      # 构建目录
```

## 核心组件

### 1. GstVideorecognition 插件

负责 GStreamer 元素实现，处理输入缓冲并输出识别结果元数据。

**GObject 属性：**

| 属性 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `unique-id` | guint | 0 | 元素唯一标识符 |
| `gpu-id` | guint | 0 | GPU 设备 ID |
| `processing-width` | gint | 64 | 模型输入宽度 |
| `processing-height` | gint | 64 | 模型输入高度 |
| `model-clip-length` | gint | 32 | 视频片段帧数 |
| `sampling-rate` | gint | 5 | 帧采样间隔 |
| `trt-engine-name` | string | "models/x3d.engine" | TensorRT 引擎路径 |
| `labels-file` | string | "labels.txt" | 类别标签文件路径 |

### 2. X3dTrt 类

封装 X3D 模型的 TensorRT 推理流程：

- 输入形状：`(1, 3, 32, 64, 64)` - NCTHW 格式
- 执行 CUDA 加速推理
- 解析输出获取最高置信度类别

### 3. Process 类

视频帧预处理模块：

- 维护帧历史缓冲区（默认最大 170 帧）
- 执行帧采样和归一化
- 转换为模型输入格式

**预处理参数：**
- 归一化均值：`[0.45, 0.45, 0.45]`
- 归一化标准差：`[0.225, 0.225, 0.225]`

### 4. RECOGNITION 结果结构

```cpp
struct RECOGNITION {
    std::string class_name;  // 类别名称
    int class_id;            // 类别 ID
    float score;             // 置信度分数
};
```

## 技术特点

- **CUDA 加速**：使用 CUDA Stream 和 GPU 内存进行推理
- **内存管理**：智能缓冲区管理，支持 dGPU 和 Jetson 平台
- **智能裁剪**：目标区域中心裁剪，支持边界 padding
- **动态标签**：支持从文件加载类别名称
- **元数据输出**：通过 `NvDsClassifierMeta` 输出分类结果

## 构建依赖

### 系统依赖

- GStreamer 1.0
- CUDA Toolkit
- NVIDIA TensorRT
- DeepStream SDK
- OpenCV

### Ubuntu/Debian 安装

```bash
sudo apt-get install libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
    libgstreamer-plugins-good1.0-dev libopencv-dev cuda-toolkit
```

## 构建方法

```bash
cd /home/tl/work/host_deepstream/deepstream-app-custom/src/gst-videorecognition
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```

构建完成后，插件库将生成至：

```
build/libgst_videorecognition.so
```

## 安装

将生成的库文件复制到 DeepStream 插件目录：

```bash
sudo cp build/libgst_videorecognition.so /opt/nvidia/deepstream/deepstream/lib/gst-plugins/
```

## 模型转换

### 准备 ONNX 模型

首先训练或下载 X3D 预训练模型，导出为 ONNX 格式。

### 转换为 TensorRT 引擎

使用提供的脚本转换模型：

```bash
cd models
# FP32 模式
./convert2trt.sh x3d.onnx x3d.engine

# FP16 模式（推荐，精度略降但速度更快）
./convert2trt.sh x3d.onnx x3d.engine fp16
```

### 准备标签文件

创建 `labels.txt`，每行一个类别名称：

```
sitting
standing
walking
running
...
```

## 集成到 DeepStream 流水线

```bash
# 编译应用
cd /home/tl/work/host_deepstream/deepstream-app-custom
make

# 运行示例
deepstream-app -c configs/deepstream_app_config.txt
```

### 插件配置示例

```ini
# deepstream_app_config.txt
[stream-muxer]
...
primary-gie-group-id=0

[primary-gie]
enable=1
model-engine-file=../../models/yolov8s.engine
# ... 其他配置

# 添加视频识别后处理
[video-recognizer]
enable=1
plugin-name=videorecognition
trt-engine-name=../../models/x3d.engine
labels-file=../../models/labels.txt
processing-width=64
processing-height=64
model-clip-length=32
sampling-rate=5
unique-id=2
gpu-id=0
```

## 工作流程

```
输入视频流 → 目标检测 → 目标跟踪 → 视频识别(X3D) → 分类元数据
```

1. 接收前级模块输出的目标 ROI 区域
2. 累积多帧视频数据到历史缓冲区
3. 按采样率提取帧并进行预处理
4. 调用 TensorRT 引擎执行 X3D 推理
5. 输出分类结果附加到 NvDsObjectMeta

## 调试与故障排除

### 常见问题

1. **CUDA 内存不足**：减小 `processing-width/height` 或 `max-history-frames`
2. **推理超时**：确认 TensorRT 引擎与模型匹配
3. **元数据不显示**：检查 `unique-id` 唯一且正确配置

### 日志调试

启用详细日志：

```bash
GST_DEBUG=3 ./your_app
```

## 许可证

本项目遵循 NVIDIA DeepStream SDK 相关许可证条款。

## 参考资料

- [NVIDIA DeepStream SDK](https://developer.nvidia.com/deepstream-sdk)
- [X3D / SlowFast Models](https://github.com/facebookresearch/SlowFast)
- [TensorRT Developer Guide](https://docs.nvidia.com/deeplearning/tensorrt/)
