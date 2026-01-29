# 导入路径修复总结

## 📋 问题概述

将代码移动到 `src/` 目录后，Python 模块导入路径出现问题。

## 🔧 修复的文件列表

### 1. 核心源代码 (`src/`)

- ✅ `src/main.py` - 改为 `from config import Config`
- ✅ `src/robot_controller.py` - 保持 `from brain.*` 和 `from body.*`
- ✅ `src/brain/llm_client.py` - 改为 `from .schemas`（相对导入）
- ✅ `src/brain/prompts.py` - 改为 `from config import Config`
- ✅ `src/body/mock_robot.py` - 改为 `from .robot_interface`（相对导入）
- ✅ `src/config_local.example.py` - 改为 `from config import RunMode`

### 2. 测试文件 (`tests/`)

- ✅ `tests/test_basic.py` - 添加 `src/` 到 `sys.path`
- ✅ `tests/compare_modes.py` - 添加项目根目录和 `src/` 到 `sys.path`

### 3. 演示文件 (`demo/`)

- ✅ `demo/demo_retry.py` - 添加项目根目录和 `src/` 到 `sys.path`
- ✅ `demo/demo_navigate.py` - 已经正确（无需修改）

### 4. 根目录文件

- ✅ `test_no_thought.py` - 添加项目根目录和 `src/` 到 `sys.path`
- ✅ `config_local_example.py` - 改为 `from config import RunMode`

## 📐 导入规范

### 规则 1：`src/` 目录内部

在 `src/` 目录内的模块之间导入：

```python
# ✅ 正确：直接导入（Python 从项目根目录运行时会自动找到 src/）
from config import Config
from brain.llm_client import LLMClient
from body.mock_robot import MockRobot

# ✅ 正确：同一包内使用相对导入
from .schemas import RobotDecision
from .robot_interface import RobotInterface
```

```python
# ❌ 错误：不要使用 src. 前缀
from src.config import Config  # ❌
```

### 规则 2：外部文件导入 `src/` 内的模块

从 `tests/`、`demo/`、根目录等导入 `src/` 内的模块：

```python
import sys
import os

# 添加项目根目录和 src 目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

# 然后正常导入
from config import Config
from robot_controller import RobotController
```

## ✅ 验证测试

所有文件已验证可以正常运行：

```bash
# 测试
✅ python3 tests/test_basic.py
✅ python3 tests/compare_modes.py
✅ python3 test_no_thought.py

# 演示
✅ python3 demo/demo_retry.py
✅ python3 demo/demo_navigate.py

# 主程序
✅ python3 src/main.py --test
✅ python3 src/main.py --demo
✅ python3 src/main.py  # 交互模式
```

## 📚 运行命令规范

从项目根目录运行所有命令：

```bash
# 进入项目根目录
cd /home/frank/rcathome/LLM

# 运行主程序
python3 src/main.py [选项]

# 运行测试
python3 tests/test_basic.py
python3 tests/compare_modes.py
python3 test_no_thought.py

# 运行演示
python3 demo/demo_retry.py
python3 demo/demo_navigate.py
```

## 🎯 关键要点

1. **从项目根目录运行** - 所有命令都应该从 `/home/frank/rcathome/LLM/` 运行
2. **src/ 内部使用简洁导入** - 不需要 `src.` 前缀
3. **外部文件需要添加路径** - 使用 `sys.path.insert()` 添加 `src/` 到路径
4. **相对导入适用于同一包** - 如 `from .schemas import ...`

## 📝 文档更新

已更新以下文档以反映新的目录结构：

- ✅ `QUICKSTART.md` - 更新了运行命令和项目结构说明
- ✅ 本文档 - 记录了所有导入修复

---

**修复完成时间**: 2026-01-29  
**状态**: ✅ 所有导入问题已解决
