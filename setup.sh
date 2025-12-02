#!/bin/bash
# CADE 项目自动配置脚本

set -e  # 遇到错误立即退出

echo "╔═══════════════════════════════════════════════════════════╗"
echo "║  🤖 CADE 环境配置脚本                                     ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 检查是否已安装 conda
if ! command -v conda &> /dev/null; then
    echo -e "${RED}❌ 未检测到 conda${NC}"
    echo ""
    echo "请先安装 Miniforge:"
    echo "  1. 下载: wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
    echo "  2. 安装: bash Miniforge3-Linux-x86_64.sh"
    echo "  3. 重启终端或运行: source ~/.bashrc"
    echo ""
    echo "详细步骤请查看 SETUP_GUIDE.md"
    exit 1
fi

echo -e "${GREEN}✓${NC} 检测到 conda: $(conda --version)"
echo ""

# 询问是否创建新环境
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
read -p "是否创建新的 conda 环境 'cade'? (y/n) " -n 1 -r
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [[ $REPLY =~ ^[Yy]$ ]]; then
    # 检查环境是否已存在
    if conda env list | grep -q "^cade "; then
        echo -e "${YELLOW}⚠️  环境 'cade' 已存在${NC}"
        read -p "是否删除并重建? (y/n) " -n 1 -r
        echo ""
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo "🗑️  删除旧环境..."
            conda env remove -n cade -y
        else
            echo "跳过创建环境"
            ENV_EXISTS=true
        fi
    fi

    if [ "$ENV_EXISTS" != "true" ]; then
        echo "📦 创建 conda 环境 'cade' (Python 3.11)..."
        conda create -n cade python=3.11 -y
        echo -e "${GREEN}✓${NC} 环境创建完成"
    fi
fi

echo ""

# 激活环境
echo "🔄 激活环境..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate cade

# 验证 Python
echo -e "${GREEN}✓${NC} Python: $(python --version)"
echo -e "${GREEN}✓${NC} Python 路径: $(which python)"
echo ""

# 安装依赖
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
read -p "是否安装项目依赖? (y/n) " -n 1 -r
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "📥 安装依赖包..."
    pip install -r requirements.txt
    echo -e "${GREEN}✓${NC} 依赖安装完成"
    echo ""
fi

# 配置文件
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [ ! -f config_local.py ]; then
    echo "📝 创建配置文件..."
    cp config_local.example.py config_local.py
    echo -e "${GREEN}✓${NC} 已创建 config_local.py"
    echo ""
    echo -e "${YELLOW}⚠️  重要提示:${NC}"
    echo "   请编辑 config_local.py 填入你的 API 密钥"
    echo "   或者安装 Ollama 使用本地模型"
    echo ""
else
    echo -e "${GREEN}✓${NC} config_local.py 已存在"
    echo ""
fi

# 运行测试
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
read -p "是否运行基础测试? (y/n) " -n 1 -r
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🧪 运行测试..."
    python tests/test_basic.py
fi

# 完成
echo ""
echo "╔═══════════════════════════════════════════════════════════╗"
echo "║  ✅ 环境配置完成！                                        ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""
echo "下一步:"
echo "  1. 激活环境: conda activate cade"
echo "  2. 配置 API: nano config_local.py"
echo "  3. 运行演示: python main.py --demo"
echo "  4. 交互模式: python main.py"
echo ""
echo "查看帮助:"
echo "  - SETUP_GUIDE.md  : 详细环境配置步骤"
echo "  - QUICKSTART.md   : 快速开始指南"
echo "  - GUIDELINE.md    : 开发路线图"
echo ""
