# Project CADE - 具身智能机器人

基于云端大脑的服务机器人开发框架

## 架构设计

```
CADE/
├── brain/          # 大脑模块（LLM交互、意图识别、决策生成）
├── body/           # 躯干模块（硬件控制、Mock/Real切换）
├── tests/          # 测试用例
├── config.py       # 配置抽象层（支持云端/本地切换）
└── main.py         # 主入口
```

## 开发阶段

### 第一阶段：PC端开发（当前）
- [x] 建立项目结构
- [ ] 配置抽象层
- [ ] LLM Client 实现
- [ ] Mock Robot 实现
- [ ] 端到端测试

### 第二阶段：Orin部署（硬件到货后）
- [ ] 部署 Ollama
- [ ] 切换为本地模型
- [ ] 对接真实 ROS 2

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 配置API（创建 config.local.py）
cp config.py config.local.py
# 编辑 config.local.py，填入你的API密钥

# 运行测试
python main.py
```

## 技术栈

- **推理引擎**: DeepSeek/DashScope (云端) → Ollama (本地)
- **基础模型**: Qwen 2.5 (3B-Instruct)
- **API标准**: OpenAI Compatible
- **数据校验**: Pydantic
- **机器人通信**: ROS 2 Humble (未来)
