# 环境配置指南

## 系统信息

- 操作系统: WSL2 (Linux)
- 架构: x86_64

## 第一步：安装 Miniforge

Miniforge 是一个轻量级的 conda 发行版，包含 conda-forge 作为默认 channel。

### 1.1 下载 Miniforge

```bash
# 下载 Miniforge 安装脚本
cd ~
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh

# 如果 wget 不可用，可以使用 curl
# curl -L -O https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
```

### 1.2 运行安装脚本

```bash
# 添加执行权限
chmod +x Miniforge3-Linux-x86_64.sh

# 运行安装（会有交互式提示）
bash Miniforge3-Linux-x86_64.sh
```

安装过程中：
- 按 Enter 开始阅读许可协议
- 输入 `yes` 接受许可
- 确认安装路径（默认 `~/miniforge3`，直接按 Enter）
- 选择 `yes` 让安装程序初始化 miniforge（会修改 `.bashrc`）

### 1.3 激活 conda

```bash
# 重新加载 shell 配置
source ~/.bashrc

# 或者直接激活 miniforge
source ~/miniforge3/bin/activate

# 验证安装
conda --version
python --version
```

## 第二步：创建项目环境

### 2.1 创建专用虚拟环境

```bash
# 进入项目目录
cd /mnt/c/Users/pc/Projects/CADE

# 创建名为 'cade' 的环境，使用 Python 3.11
conda create -n cade python=3.11 -y

# 激活环境
conda activate cade

# 验证
which python
python --version
```

### 2.2 安装项目依赖

```bash
# 确保在 cade 环境中（命令行前缀应显示 (cade)）
pip install -r requirements.txt

# 验证安装
pip list
```

## 第三步：配置 API（可选）

### 方案 A：使用云端 API（需要网络和密钥）

```bash
# 复制配置示例
cp config_local.example.py config_local.py

# 使用你喜欢的编辑器编辑（vim/nano/code）
nano config_local.py
# 或
code config_local.py
```

在 `config_local.py` 中取消注释并填写：

```python
from config import RunMode

MODE = RunMode.CLOUD
CLOUD_API_KEY = "sk-your-actual-deepseek-key-here"
```

获取 DeepSeek API 密钥：
1. 访问 https://platform.deepseek.com/
2. 注册/登录
3. 创建 API Key
4. 新用户通常有免费额度

### 方案 B：使用本地 Ollama（免费，离线）

```bash
# 安装 Ollama
curl -fsSL https://ollama.com/install.sh | sh

# 拉取推荐模型（约 2GB）
ollama pull qwen2.5:3b

# 验证
ollama list

# 配置为本地模式
echo 'from config import RunMode
MODE = RunMode.LOCAL' > config_local.py
```

## 第四步：运行测试

### 4.1 基础模块测试（无需API）

```bash
# 激活环境
conda activate cade

# 运行测试
python tests/test_basic.py
```

应该看到类似输出：
```
=== 测试 Config ===
✓ 配置模块正常
...
✅ 所有测试通过！
```

### 4.2 端到端测试（需要API/Ollama）

```bash
# 演示模式（完整服务流程）
python main.py --demo

# 测试模式（多个测试用例）
python main.py --test

# 交互模式
python main.py
```

## 常用命令

```bash
# 激活项目环境
conda activate cade

# 退出环境
conda deactivate

# 查看所有环境
conda env list

# 更新依赖
pip install --upgrade -r requirements.txt

# 删除环境（如需重建）
conda env remove -n cade
```

## 故障排查

### 问题 1: conda 命令找不到

```bash
# 手动激活 miniforge
source ~/miniforge3/bin/activate

# 或添加到 PATH
export PATH="$HOME/miniforge3/bin:$PATH"
```

### 问题 2: pip 安装失败

```bash
# 更新 pip
pip install --upgrade pip

# 使用国内镜像（可选，加速）
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 问题 3: API 调用失败

检查：
1. `config_local.py` 是否存在且配置正确
2. API 密钥是否有效
3. 网络是否正常

```bash
# 查看配置
python -c "from config import Config; print(Config.get_llm_config())"
```

### 问题 4: Ollama 模型下载慢

```bash
# 使用代理（如果有）
export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890

ollama pull qwen2.5:3b
```

## 下一步

环境配置完成后：
1. 阅读 `QUICKSTART.md` 开始使用
2. 查看 `GUIDELINE.md` 了解开发路线图
3. 运行 `python main.py` 开始体验

## 自动化配置脚本（可选）

创建一个一键配置脚本：

```bash
cat > setup.sh << 'EOF'
#!/bin/bash
set -e

echo "🚀 开始配置 CADE 环境..."

# 创建 conda 环境
conda create -n cade python=3.11 -y

# 激活环境
source ~/miniforge3/etc/profile.d/conda.sh
conda activate cade

# 安装依赖
pip install -r requirements.txt

# 复制配置示例
if [ ! -f config_local.py ]; then
    cp config_local.example.py config_local.py
    echo "⚠️  请编辑 config_local.py 填入你的 API 密钥"
fi

# 运行测试
echo "🧪 运行基础测试..."
python tests/test_basic.py

echo "✅ 环境配置完成！"
echo "下一步: 编辑 config_local.py，然后运行 python main.py"
EOF

chmod +x setup.sh
```

使用方法：
```bash
./setup.sh
```
