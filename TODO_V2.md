# CADE 项目迭代计划 V2.0

> **当前状态**: 基础架构完成，Mock模式运行良好
> **目标**: 从原型系统升级为可部署的应用级机器人系统
> **时间规划**: 12-16周（分4个阶段）

---

## 📊 当前项目评估

### ✅ 已完成 (约 2700 行代码)
- [x] **Brain模块**: LLM客户端、Prompt工程、Schema定义、ReAct决策流程
- [x] **Body模块**: Mock Robot、抽象接口、状态机基础
- [x] **Controller**: 完整的感知-决策-执行循环
- [x] **配置层**: 支持云端/本地模式切换
- [x] **多种运行模式**: 交互/测试/演示
- [x] **基础文档**: README、QUICKSTART、GUIDELINE

### ⚠️ 待完善方向
1. **稳定性**: 缺少完整的错误处理和恢复机制
2. **硬件集成**: 尚未对接真实ROS 2系统
3. **感知能力**: 缺少视觉、语音等多模态输入
4. **测试覆盖**: 单元测试不足
5. **用户体验**: 缺少可视化界面和监控工具
6. **性能优化**: 响应速度、本地部署待优化

---

## 🎯 迭代路线图

### **第一阶段：稳定基座** (3-4周)
**目标**: 让系统更健壮、可测试、可维护

#### 1.1 测试框架 (Priority: 🔴 High)
```
tests/
├── unit/
│   ├── test_schemas.py          # 数据模型验证
│   ├── test_llm_client.py       # LLM客户端测试
│   ├── test_robot_controller.py # 控制器逻辑测试
│   └── test_prompts.py          # Prompt生成测试
├── integration/
│   ├── test_end_to_end.py       # 端到端场景测试
│   └── test_conversation.py     # 多轮对话测试
├── performance/
│   └── benchmark.py             # 性能基准测试
└── fixtures/
    └── test_data.json           # 测试数据集
```

**任务清单**:
- [ ] 安装 pytest, pytest-cov, pytest-asyncio
- [ ] 编写单元测试（目标覆盖率 >80%）
- [ ] 建立回归测试集（50个标准场景）
- [ ] 添加性能benchmark（响应时间、token使用）
- [ ] 配置 GitHub Actions CI

#### 1.2 错误处理增强 (Priority: 🔴 High)
```python
# 新增模块
brain/
└── exceptions.py       # 自定义异常类型

body/
└── recovery.py         # 自动恢复策略

utils/
├── retry.py           # 重试装饰器
└── circuit_breaker.py # 熔断器
```

**任务清单**:
- [ ] 定义异常分类体系（网络/硬件/逻辑错误）
- [ ] 实现重试机制（exponential backoff）
- [ ] 添加熔断器（防止级联失败）
- [ ] 设计降级策略（如LLM失败时使用规则引擎）
- [ ] 实现动作超时和自动中断

#### 1.3 日志和监控 (Priority: 🟡 Medium)
```python
# 改进现有日志
utils/
├── logger.py          # 结构化日志配置
└── metrics.py         # 性能指标收集

# 输出格式
logs/
├── robot_YYYYMMDD.log      # 每日日志
├── error_YYYYMMDD.log      # 错误日志
└── performance_YYYYMMDD.json # 性能数据
```

**任务清单**:
- [ ] 配置 loguru 结构化日志
- [ ] 添加日志级别和过滤
- [ ] 实现性能指标收集（latency, token usage, success rate）
- [ ] 日志轮转和归档策略
- [ ] （可选）集成 Prometheus + Grafana

#### 1.4 任务队列和状态机 (Priority: 🟡 Medium)
```python
brain/
└── task_manager.py    # 任务队列和规划器

# 支持复杂任务分解
User: "把苹果放到桌子上"
→ Task Queue: [search(apple), navigate(apple), pick(apple), navigate(table), place(table)]
```

**任务清单**:
- [ ] 实现任务队列（支持暂停/恢复）
- [ ] 添加任务优先级管理
- [ ] 设计状态机（支持中断和异常处理）
- [ ] 实现任务执行历史记录
- [ ] 支持多步骤任务的断点续传

---

### **第二阶段：硬件对接** (4-6周)
**目标**: 在真实机器人上运行

#### 2.1 ROS 2 接口实现 (Priority: 🔴 High)
```
body/
├── real_robot.py          # ROS 2 实现
├── ros_bridge.py          # ROS消息转换
└── calibration/
    ├── camera_calib.yaml  # 相机标定
    └── tf_tree.yaml       # 坐标系变换
```

**技术栈**:
- ROS 2 Humble
- rclpy (Python ROS客户端)
- Nav2 (导航栈)
- MoveIt2 (抓取规划)

**任务清单**:
- [ ] 搭建ROS 2开发环境（Docker或虚拟机）
- [ ] 实现 `RealRobot` 类（继承 `RobotInterface`）
- [ ] 对接导航：`navigate()` → Nav2 ActionClient
- [ ] 对接抓取：`pick()` → MoveIt2 规划器
- [ ] 实现传感器数据订阅（LaserScan, PointCloud）
- [ ] 坐标系转换（map ↔ base_link ↔ camera）
- [ ] 编写ROS 2 launch文件

#### 2.2 视觉模块集成 (Priority: 🔴 High)
```
perception/
├── __init__.py
├── detector.py           # 目标检测
├── segmentation.py       # 实例分割
└── models/
    ├── yolo_config.yaml
    └── clip_config.yaml
```

**技术选型**:
- 目标检测: YOLOv8 / YOLOv10
- 语义理解: CLIP / GroundingDINO
- 位姿估计: FoundationPose

**任务清单**:
- [ ] 集成YOLOv8进行物体检测
- [ ] 实现 `search()` 的视觉实现
- [ ] 3D位姿估计（RGB-D → 物体坐标）
- [ ] 物体跟踪（Kalman滤波）
- [ ] 与ROS 2集成（发布Detection消息）

#### 2.3 导航和建图 (Priority: 🟡 Medium)
```
mapping/
├── semantic_map.py       # 语义地图
└── location_manager.py   # 地点管理

# 数据格式
maps/
├── office_map.yaml       # 占用栅格地图
└── semantic_locations.json  # 语义标签
{
  "kitchen": {"coords": [2.5, 3.0, 0.0], "type": "room"},
  "table": {"coords": [1.0, 1.5, 0.8], "type": "furniture"}
}
```

**任务清单**:
- [ ] 使用SLAM构建地图（Cartographer/SLAM Toolbox）
- [ ] 建立语义地图（room → coords映射）
- [ ] 实现语义导航（"去厨房" → 查表 → Nav2）
- [ ] 动态障碍物避障
- [ ] 充电桩/Home位置标定

#### 2.4 机械臂和抓取 (Priority: 🟡 Medium)
```
manipulation/
├── grasp_planner.py     # 抓取规划
└── gripper_control.py   # 夹爪控制
```

**任务清单**:
- [ ] MoveIt2抓取规划
- [ ] 夹爪开合控制
- [ ] 碰撞检测
- [ ] 力反馈控制（避免损坏物体）

---

### **第三阶段：智能增强** (3-4周)
**目标**: 多模态交互，更智能的决策

#### 3.1 多模态LLM集成 (Priority: 🔴 High)
```python
brain/
├── vlm_client.py         # 视觉语言模型客户端
└── multimodal_prompt.py  # 多模态Prompt

# 使用场景
User: "这是什么？" [同时发送相机图像]
VLM: "这是一个红色的苹果"
```

**技术选型**:
- Qwen2-VL (本地部署)
- GPT-4V / Claude 3.5 (云端备选)

**任务清单**:
- [ ] 集成VLM API客户端
- [ ] 实现图像+文本的联合Prompt
- [ ] 支持"看图识物"场景
- [ ] 视觉问答（VQA）
- [ ] 场景理解（场景描述生成）

#### 3.2 语音交互 (Priority: 🟡 Medium)
```
voice/
├── asr.py               # 语音识别
├── tts.py               # 语音合成
└── wake_word.py         # 唤醒词检测
```

**技术选型**:
- ASR: Whisper / FunASR
- TTS: Edge-TTS / PaddleSpeech
- 唤醒词: Porcupine

**任务清单**:
- [ ] 集成Whisper进行语音识别
- [ ] 集成TTS引擎
- [ ] 实现唤醒词检测（"你好LARA"）
- [ ] 语音端点检测（VAD）
- [ ] 噪声抑制

#### 3.3 本地模型部署 (Priority: 🔴 High)
```bash
# Jetson Orin 部署方案
├── Ollama (推理引擎)
│   └── Qwen2.5:3B (量化版)
├── TensorRT (加速)
└── vLLM (可选，batch推理)
```

**任务清单**:
- [ ] 在Orin上安装Ollama
- [ ] 部署Qwen2.5-3B-Instruct-Q4
- [ ] 性能测试（latency, throughput）
- [ ] 量化优化（INT8/INT4）
- [ ] 切换config.py为LOCAL模式
- [ ] 对比云端/本地性能差异

#### 3.4 记忆和上下文管理 (Priority: 🟢 Low)
```python
brain/
├── memory.py            # 长期记忆
└── context_manager.py   # 上下文窗口管理

# 支持跨对话记忆
Session 1: "我叫张三"
Session 2: "我是谁？" → "您是张三"
```

**任务清单**:
- [ ] 实现向量数据库（Chroma/FAISS）
- [ ] 存储对话历史
- [ ] 实现RAG（检索增强生成）
- [ ] 上下文窗口滑动策略
- [ ] 用户偏好记忆

---

### **第四阶段：用户体验** (2-3周)
**目标**: 可视化、易用、可演示

#### 4.1 Web监控面板 (Priority: 🔴 High)
```
web/
├── backend/
│   └── server.py        # FastAPI服务
├── frontend/
│   ├── index.html
│   ├── app.js          # Vue.js / React
│   └── styles.css
└── static/
    └── assets/
```

**功能设计**:
- 实时机器人状态显示
- 相机视频流
- 地图可视化（ROS 2地图显示）
- 对话历史
- 性能监控图表
- 远程控制（发送指令）

**任务清单**:
- [ ] 搭建FastAPI后端
- [ ] 实现WebSocket实时通信
- [ ] 前端UI开发（推荐Vue 3）
- [ ] 视频流转发（WebRTC/MJPEG）
- [ ] ROS 2地图可视化（rosbridge）
- [ ] 移动端适配

#### 4.2 RViz可视化 (Priority: 🟡 Medium)
```bash
# RViz配置
rviz/
└── robot_view.rviz     # 预配置的RViz布局

# 显示内容
- 机器人模型（URDF）
- 地图（OccupancyGrid）
- 激光雷达数据
- 相机图像
- 检测框（BoundingBox）
- 导航路径
```

**任务清单**:
- [ ] 配置RViz显示
- [ ] 添加自定义Marker（检测结果）
- [ ] 路径可视化
- [ ] 交互式导航（点击地图发送目标）

#### 4.3 移动端App (Priority: 🟢 Low)
```
mobile/
└── flutter_app/        # 跨平台App
    ├── lib/
    └── pubspec.yaml
```

**功能**:
- 语音控制
- 远程查看
- 快捷指令
- 推送通知

**任务清单**:
- [ ] Flutter项目初始化
- [ ] 接入Web API
- [ ] 实现语音输入
- [ ] 视频流显示

#### 4.4 文档和演示 (Priority: 🟡 Medium)
```
docs/
├── API.md              # API文档
├── DEPLOYMENT.md       # 部署指南
├── ARCHITECTURE.md     # 架构文档
└── videos/
    └── demo.mp4        # 演示视频
```

**任务清单**:
- [ ] 补充API文档（Swagger自动生成）
- [ ] 编写部署教程
- [ ] 录制演示视频
- [ ] 制作PPT演示材料

---

## 🔧 技术债务清理

### 代码质量
- [ ] 添加类型注解（mypy检查）
- [ ] 代码格式化（black, isort）
- [ ] Lint检查（pylint, flake8）
- [ ] 代码审查流程

### 安全性
- [ ] API密钥加密存储
- [ ] 用户输入验证
- [ ] SQL注入防护（如果使用数据库）
- [ ] HTTPS部署
- [ ] 安全模式（限制危险动作）

### 性能优化
- [ ] 异步化（asyncio）
- [ ] 请求缓存
- [ ] 数据库查询优化
- [ ] 内存泄漏检查

---

## 📈 里程碑和验收标准

### Milestone 1: 稳定基座 (Week 3-4)
**验收标准**:
- [ ] 测试覆盖率 >80%
- [ ] CI/CD流程运行正常
- [ ] 错误恢复机制完善
- [ ] 日志系统完整

### Milestone 2: 硬件对接 (Week 8-10)
**验收标准**:
- [ ] 在真实机器人上完成导航任务
- [ ] 视觉检测准确率 >85%
- [ ] 抓取成功率 >70%
- [ ] ROS 2集成无故障

### Milestone 3: 智能增强 (Week 12-13)
**验收标准**:
- [ ] 支持视觉问答
- [ ] 语音交互流畅
- [ ] 本地模型推理延迟 <2s
- [ ] 多轮对话理解准确

### Milestone 4: 用户体验 (Week 15-16)
**验收标准**:
- [ ] Web面板功能完整
- [ ] 演示视频制作完成
- [ ] 文档齐全
- [ ] 用户测试反馈良好

---

## 🚀 快速启动建议

### 立即开始（本周）
1. **搭建测试框架** - 安装pytest，编写前10个测试用例
2. **设计ROS 2接口** - 绘制接口UML图，定义消息格式
3. **创建Web面板雏形** - FastAPI + 简单前端

### 优先级排序
```
🔴 High Priority (必须):
- 测试框架
- 错误处理
- ROS 2接口
- 视觉模块
- 本地模型部署
- Web监控面板

🟡 Medium Priority (重要):
- 日志系统
- 任务队列
- 导航建图
- 语音交互
- RViz可视化

🟢 Low Priority (可选):
- 移动端App
- 记忆系统
- 高级功能
```

---

## 📝 附录

### 技术栈总览
```yaml
Language: Python 3.10+
Framework:
  Backend: FastAPI, ROS 2 Humble
  Frontend: Vue 3 / React
  Mobile: Flutter (可选)

AI/ML:
  LLM: Qwen2.5-3B (Ollama)
  VLM: Qwen2-VL
  Vision: YOLOv8, CLIP
  Speech: Whisper, Edge-TTS

Robotics:
  Navigation: Nav2
  Manipulation: MoveIt2
  SLAM: Cartographer

Infrastructure:
  Test: pytest, pytest-cov
  Log: loguru
  Monitor: Prometheus (可选)
  DB: SQLite / PostgreSQL (可选)
```

### 参考资源
- [ROS 2 Humble Docs](https://docs.ros.org/en/humble/)
- [Nav2 Documentation](https://navigation.ros.org/)
- [Ollama Documentation](https://ollama.ai/docs)
- [FastAPI Tutorial](https://fastapi.tiangolo.com/)

---

**最后更新**: 2025-12-05
**版本**: V2.0
**维护者**: Project CADE Team
