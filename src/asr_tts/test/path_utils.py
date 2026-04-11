"""
路径工具 - 解决 sherpa-onnx C++层不支持中文路径的问题
策略：
1. 先尝试 Windows API GetShortPathNameW 获取短路径
2. 如果仍然包含非ASCII字符，使用 subst 创建虚拟驱动器映射
"""

import os
import shutil
import subprocess
from pathlib import Path

# 尝试使用 ctypes 调用 Windows API
try:
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
    GetShortPathNameW = kernel32.GetShortPathNameW
    GetShortPathNameW.argtypes = [wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD]
    GetShortPathNameW.restype = wintypes.DWORD
    _WINAPI_AVAILABLE = True
except Exception:
    _WINAPI_AVAILABLE = False


def is_ascii(s: str) -> bool:
    """检查字符串是否全为ASCII"""
    try:
        s.encode('ascii')
        return True
    except UnicodeEncodeError:
        return False


def get_short_path_api(path: str) -> str:
    """使用Windows API获取短路径"""
    if not _WINAPI_AVAILABLE:
        return path

    resolved = str(Path(path).resolve())
    output_buf = ctypes.create_unicode_buffer(1024)
    ret = GetShortPathNameW(resolved, output_buf, 1024)

    if ret == 0 or ret >= 1024:
        return path

    return output_buf.value


# 全局subst映射缓存
_subst_mappings = {}


def ensure_ascii_path(path: str) -> str:
    """
    确保返回一个纯ASCII的文件路径。
    如果原始路径包含非ASCII字符，使用 subst 创建虚拟驱动器。
    """
    if is_ascii(path):
        return path

    # 先尝试API短路径
    short = get_short_path_api(path)
    if is_ascii(short):
        return short

    # API短路径仍包含非ASCII，使用subst
    resolved = Path(path).resolve()

    # 找到第一个非ASCII祖先目录，对它做subst
    parts = resolved.parts
    ascii_parts = []
    subst_target = None
    subst_idx = None

    for i, part in enumerate(parts):
        if not is_ascii(part):
            # 对这个部分之前的完整路径做subst
            subst_idx = i
            break
        ascii_parts.append(part)

    if subst_idx is None or subst_idx <= 1:
        print(f"[WARNING] Cannot create subst mapping for root-level path: {path}")
        return path

    # 包含第一个非ASCII目录及其之前的所有部分作为subst目标
    target_dir = Path(*parts[:subst_idx + 1])
    target_key = str(target_dir).lower()

    if target_key not in _subst_mappings:
        # 寻找可用驱动器字母
        for letter in 'ZYXWVUTSRQPONMLKJIHGFEDCBA':
            drive = f"{letter}:"
            result = subprocess.run(
                ["subst", drive, str(target_dir)],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                _subst_mappings[target_key] = drive
                print(f"[INFO] Created subst: {drive} -> {target_dir}")
                break
        else:
            print(f"[ERROR] No available drive letters for subst mapping!")
            return path

    drive = _subst_mappings[target_key]

    # 用驱动器字母替换前缀（跳过已映射的部分）
    remaining_parts = list(parts[subst_idx + 1:])
    # 驱动器后需要加反斜杠作为根目录
    new_path = Path(drive + "\\", *remaining_parts) if remaining_parts else Path(drive)
    # 注意：不要调用resolve()，否则会展开回原中文路径！
    return str(new_path)


def cleanup_subst():
    """清理所有创建的subst映射"""
    global _subst_mappings
    for target, drive in _subst_mappings.items():
        try:
            subprocess.run(["subst", drive, "/D"], capture_output=True, timeout=5)
            print(f"[INFO] Removed subst: {drive}")
        except Exception as e:
            print(f"[WARNING] Failed to remove subst {drive}: {e}")
    _subst_mappings = {}


def resolve_model_dir(model_dir: str) -> str:
    """解析模型目录，确保返回的路径可以被C++层正确读取"""
    return ensure_ascii_path(str(model_dir))


def get_short_path(path: str) -> str:
    """对外接口 - 确保路径为纯ASCII"""
    return ensure_ascii_path(path)
