# 快速开始指南

## 第一步：安装依赖

```bash
# 创建虚拟环境（推荐）
python -m venv venv

# 激活虚拟环境
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

## 第二步：配置 API 密钥

### 方案 A：使用云端 API（推荐用于开发）

1. 注册DeepSeek账号并获取API密钥
   - 访问: https://platform.deepseek.com/
   - 创建 API Key

2. 配置密钥
```bash
# 复制配置示例
cp config_local.example.py config_local.py

# 编辑 config_local.py，填入你的API密钥
# 取消注释相关行并填写：
# MODE = RunMode.CLOUD
# CLOUD_API_KEY = "sk-your-actual-key-here"
```

### 方案 B：使用本地 Ollama（无需API密钥）

```bash
# 1. 安装 Ollama
curl -fsSL https://ollama.com/install.sh | sh

# 2. 拉取模型
ollama pull qwen2.5:3b

# 3. 配置为本地模式
# 在 config_local.py 中设置:
# MODE = RunMode.LOCAL
```

## 第三步：运行测试

```bash
# 运行基础模块测试（不需要API）
python tests/test_basic.py

# 运行端到端测试（需要API）
python main.py --test

# 运行演示
python main.py --demo
```

## 第四步：交互式使用

```bash
# 启动交互模式
python main.py

# 然后你可以输入：
# "你好"
# "去厨房"
# "帮我找苹果"
# "status" - 查看机器人状态
# "stats" - 查看统计信息
# "quit" - 退出
```

## 项目结构说明

```
CADE/
├── brain/              # 🧠 大脑模块
│   ├── llm_client.py   # LLM 调用客户端
│   ├── prompts.py      # 系统提示词
│   └── schemas.py      # 数据模型（动作定义）
├── body/               # 🦾 躯干模块
│   ├── robot_interface.py  # 机器人接口
│   └── mock_robot.py   # Mock 实现
├── tests/              # 🧪 测试
│   └── test_basic.py
├── config.py           # ⚙️ 配置（默认值）
├── config_local.py     # ⚙️ 本地配置（你的密钥）
├── robot_controller.py # 🎮 主控制器
└── main.py             # 🚀 入口程序
```

## 常见问题

### Q: 没有API密钥怎么办？

A: 使用本地 Ollama 方案（见方案B），完全免费且离线运行。

### Q: API调用失败？

A: 检查：
1. API密钥是否正确
2. 网络是否正常
3. 账户余额是否充足

### Q: 如何切换模型？

A: 在 `config_local.py` 中修改：
- 云端: 修改 `CLOUD_MODEL`
- 本地: 修改 `LOCAL_MODEL`

### Q: 如何部署到真实机器人？

A: 见 `GUIDELINE.md` 第二阶段说明：
1. 在 Orin 上部署 Ollama
2. 修改 `MODE = RunMode.LOCAL`
3. 替换 `MockRobot` 为 `RealRobot`（对接ROS）

## 下一步

- 查看 `GUIDELINE.md` 了解完整开发计划
- 查看 `TODO.md` 了解架构设计思路
- 阅读代码中的注释和文档字符串
- 根据需求自定义提示词（`brain/prompts.py`）
- 添加新的动作类型（`brain/schemas.py`）

## 获取帮助

遇到问题？
1. 检查各个模块的 `if __name__ == "__main__"` 部分的测试代码
2. 运行 `python -m pydoc <module_name>` 查看文档
3. 查看代码注释

祝开发顺利！ 🚀
