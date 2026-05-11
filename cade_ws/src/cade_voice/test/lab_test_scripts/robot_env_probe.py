#!/usr/bin/env python3
"""
系统环境探测脚本 - robot_env_probe.py

一键收集机器人硬件/软件环境信息，用于确认部署条件。
输出 JSON 报告 + 终端表格。

用法:
    # 基本探测
    python robot_env_probe.py

    # 指定输出目录和文件名
    python robot_env_probe.py --output-dir ./reports --output-file my_robot_info.json

    # 只显示不保存文件
    python robot_env_probe.py --no-save
"""

import json
import os
import platform
import subprocess
import sys
import time
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path


def run_cmd(cmd, timeout=10):
    """执行 shell 命令，返回 (returncode, stdout, stderr)"""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=timeout
        )
        return result.returncode, result.stdout.strip(), result.stderr.strip()
    except subprocess.TimeoutExpired:
        return -1, "", "Command timed out"
    except Exception as e:
        return -1, "", str(e)


def probe_gpu():
    """探测 GPU 信息"""
    gpu_info = {"available": False}

    code, stdout, _ = run_cmd("nvidia-smi --query-gpu=name,memory.total,driver_version,cuda_version "
                              "--format=csv,noheader,nounits", timeout=15)
    if code == 0 and stdout:
        gpu_info["available"] = True
        lines = stdout.strip().split("\n")
        if len(lines) >= 1:
            parts = [p.strip() for p in lines[0].split(",")]
            gpu_info["name"] = parts[0] if len(parts) > 0 else "Unknown"
            gpu_info["memory_total_mb"] = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None
            gpu_info["driver_version"] = parts[2] if len(parts) > 2 else "Unknown"
            try:
                cuda_ver = int(parts[3]) if len(parts) > 3 else 0
                gpu_info["cuda_version"] = f"{cuda_ver // 1000}.{(cuda_ver % 1000) // 10}"
            except (ValueError, IndexError):
                gpu_info["cuda_version"] = "Unknown"

        # 获取当前 GPU 使用情况
        code2, usage_out, _ = run_cmd(
            "nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,"
            "temperature.gpu,power.draw --format=csv,noheader,nounits", timeout=10
        )
        if code2 == 0 and usage_out:
            uparts = [p.strip() for p in usage_out.split("\n")[0].split(",")]
            gpu_info["gpu_utilization_pct"] = uparts[0] if len(uparts) > 0 else "N/A"
            gpu_info["memory_utilization_pct"] = uparts[1] if len(uparts) > 1 else "N/A"
            gpu_info["memory_used_mb"] = uparts[2] if len(uparts) > 2 else "N/A"
            gpu_info["temperature_c"] = uparts[3] if len(uparts) > 3 else "N/A"
            gpu_info["power_watts"] = uparts[4] if len(uparts) > 4 else "N/A"

    return gpu_info


def probe_cpu_memory():
    """探测 CPU 和内存信息"""
    info = {}

    if os.path.exists("/proc/cpuinfo"):
        with open("/proc/cpuinfo") as f:
            cpuinfo = f.read()
        # CPU 型号
        import re
        m = re.search(r"model name\s*:\s*(.+)", cpuinfo)
        if m:
            info["model_name"] = m.group(1).strip()
        # 核心数
        cores = re.findall(r"processor\s*:\s*\d+", cpuinfo)
        info["logical_cores"] = len(cores)
        # 物理核心
        siblings = re.search(r"siblings\s*:\s*(\d+)", cpuinfo)
        cpu_cores = re.search(r"cpu cores\s*:\s*(\d+)", cpuinfo)
        if siblings and cpu_cores:
            info["physical_cores"] = int(cpu_cores.group(1))
        else:
            info["physical_cores"] = info.get("logical_cores", "Unknown")

        # 内存
        if os.path.exists("/proc/meminfo"):
            with open("/proc/meminfo") as f:
                meminfo = f.read()
            def extract_mem_kb(label):
                m = re.search rf"{label}:\s+(\d+)\s+kB" , meminfo
                return int(m.group(1)) // 1024 if m else None

            info["total_memory_mb"] = extract_mem_kb("MemTotal")
            info["free_memory_mb"] = extract_mem_kb("MemFree")
            info["available_memory_mb"] = extract_mem_kb("MemAvailable")

    elif platform.system() == "Windows":
        import wmi
        try:
            c = wmi.WMI()
            for proc in c.Win32_Processor():
                info["model_name"] = proc.Name.strip() if proc.Name else "Unknown"
                info["physical_cores"] = proc.NumberOfCores
                info["logical_cores"] = proc.NumberOfLogicalProcessors
                break
            for mem in c.OperatingSystem():
                info["total_memory_mb"] = round(int(mem.TotalVisibleMemorySize) / 1024)
                info["free_memory_mb"] = round(int(mem.FreePhysicalMemory) / 1024)
                break
        except ImportError:
            info["_note"] = "Install 'wmi' package for Windows hardware info: pip install wmi"
            info["model_name"] = platform.processor() or "Unknown"
    else:
        info["model_name"] = platform.processor() or "Unknown"
        info["total_memory_mb"] = None

    return info


def probe_os():
    """操作系统信息"""
    return {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "architecture": platform.architecture()[0],
        "hostname": platform.node(),
    }


def probe_audio_devices():
"""音频设备列表"""
    devices = []
    try:
        import sounddevice as sd
        devs = sd.query_devices(device=None)
        for i, d in enumerate(devs):
            devices.append({
                "id": i,
                "name": d["name"],
                "max_input_channels": d["max_input_channels"],
                "max_output_channels": d["max_output_channels"],
                "default_samplerate": d["default_samplerate"],
                "hostapi": d.get("hostapi", 0),
            })
    except ImportError:
        devices.append({"error": "sounddevice not installed"})
    except Exception as e:
        devices.append({"error": str(e)})
    return devices


def probe_docker():
    """Docker / Container Toolkit 信息"""
    docker_info = {}

    code, _, _ = run_cmd("docker --version", timeout=5)
    docker_info["docker_installed"] = code == 0
    if code == 0:
        docker_info["docker_version"] = _

    code2, out2, _ = run_cmd("docker ps --format '{{.Names}}'", timeout=5)
    docker_info["running_containers"] = out2.split("\n") if code2 == 0 and out2 else []

    code3, out3, _ = run_cmd("nvidia-container-cli list", timeout=5)
    docker_info["nvidia_container_toolkit"] = code3 == 0

    return docker_info


def probe_sherpa_onnx():
    """sherpa-onnx Python 包版本"""
    pkg_info = {}
    try:
        import sherpa_onnx
        pkg_info["installed"] = True
        pkg_info["version"] = getattr(sherpa_onnx, "__version__", "unknown")
    except ImportError:
        pkg_info["installed"] = False
    return pkg_info


def probe_disk_space(target_path="/"):
    """磁盘空间"""
    disk_info = {}
    try:
        usage = shutil.disk_usage(target_path or ".")
        disk_info = {
            "path": target_path or ".",
            "total_gb": round(usage.total / (1024**3), 2),
            "used_gb": round(usage.used / (1024**3), 2),
            "free_gb": round(usage.free / (1024**3), 2),
            "percent_used": round(usage.used / usage.total * 100, 1),
        }
    except Exception as e:
        disk_info["error"] = str(e)
    return disk_info


def print_table(data, title=""):
    """格式化打印表格"""
    if title:
        print(f"\n{'='*60}")
        print(f"  {title}")
        print('='*60)

    if isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, dict):
                print(f"\n  [{k}]")
                for k2, v2 in v.items():
                    print(f"    {k2}: {v2}")
            elif isinstance(v, list):
                print(f"  {k}: {', '.join(str(x) for x in v[:5])}" + (f" (+{len(v)-5} more)" if len(v) > 5 else ""))
            else:
                print(f"  {k}: {v}")


def main():
    parser = ArgumentParser(description="Robot Environment Probe - 收集机器人硬件/软件环境信息")
    parser.add_argument("--output-dir", default=None, help="输出目录 (默认: 当前目录)")
    parser.add_argument("--output-file", default="robot_env_report.json", help="输出JSON文件名")
    parser.add_argument("--no-save", action="store_true", help="只打印，不保存文件")
    args = parser.parse_args()

    print("=" * 60)
    print("  Robot Environment Probe v1.0")
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    report = {
        "timestamp": datetime.now().isoformat(),
        "probe_version": "1.0",
    }

    # ---- GPU ----
    print("\n[*] Probing GPU...")
    report["gpu"] = probe_gpu()
    print_table(report["gpu"], "GPU Info")

    # ---- CPU & Memory ----
    print("\n[*] Probing CPU & Memory...")
    report["cpu_memory"] = probe_cpu_memory()
    print_table(report["cpu_memory"], "CPU & Memory")

    # ---- OS ----
    print("\n[*] Probing OS...")
    report["os"] = probe_os()
    print_table(report["os"], "OS Info")

    # ---- Audio Devices ----
    print("\n[*] Probing Audio Devices...")
    report["audio_devices"] = probe_audio_devices()
    print(f"  Found {len(report['audio_devices'])} audio device(s)")
    for dev in report["audio_devices"]:
        if "error" not in dev:
            inp = f"input:{dev['max_input_channels']}" if dev["max_input_channels"] > 0 else ""
            outp = f"output:{dev['max_output_channels']}" if dev["max_output_channels"] > 0 else ""
            print(f"    [{dev['id']}] {dev['name']} ({inp} {outp}) @ {dev['default_samplerate']}Hz")

    # ---- Docker ----
    print("\n[*] Probing Docker...")
    report["docker"] = probe_docker()
    print_table(report["docker"], "Docker / NVIDIA Container Toolkit")

    # ---- sherpa-onnx ----
    print("\n[*] Checking sherpa-onnx...")
    report["sherpa_onnx"] = probe_sherpa_onnx()
    print_table(report["sherpa_onnx"], "sherpa-onnx Package")

    # ---- Disk ----
    print("\n[*] Checking Disk Space...")
    target_dir = args.output_dir or "."
    report["disk"] = probe_disk_space(target_dir)
    print_table(report["disk"], f"Disk Space ({target_dir or '.'})")

    # ---- Quick Compatibility Check ----
    print("\n" + "=" * 60)
    print("  Deployment Readiness Summary")
    print("=" * 60)

    checks = [
        ("GPU Available", report["gpu"].get("available", False)),
        ("Audio Input Device", any(d.get("max_input_channels", 0) > 0 for d in report["audio_devices"] if "error" not in d)),
        ("Audio Output Device", any(d.get("max_output_channels", 0) > 0 for d in report["audio_devices"] if "error" not in d)),
        ("sherpa-onnx Installed", report["sherpa_onnx"].get("installed", False)),
        ("Disk Free > 5GB", (report["disk"].get("free_gb", 0) or 0) > 5),
    ]

    all_pass = True
    for name, passed in checks:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status} {name}")
        if not passed:
            all_pass = False

    print(f"\n  Overall: {'READY' if all_pass else 'ISSUES DETECTED'}")

    # ---- Save Report ----
    if not args.no_save:
        output_dir = Path(args.output_dir) if args.output_dir else Path(".")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / args.output_file
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        print(f"\n[OK] Report saved to: {output_path.resolve()}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    import shutil
    sys.exit(main())
