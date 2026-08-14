# -*- coding: utf-8 -*-
"""硬件检测模块 — 纯 stdlib 实现, 零第三方依赖。

检测: OS / CPU / 内存 / GPU (NVIDIA 优先, 回退 WMI) / 磁盘空间。
在安装程序里用, 打包成 exe 后不需要装任何依赖。
"""
import ctypes
import json
import os
import platform
import shutil
import struct
import subprocess
import sys
import re


def _ps_json(query):
    """PowerShell Get-CimInstance -> JSON, 返回 list[dict] 或 dict。"""
    cmd = ["powershell", "-NoProfile", "-Command",
           f"(Get-CimInstance {query}) | ConvertTo-Json -Compress"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=20,
                           encoding="utf-8", errors="replace",
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if r.returncode != 0 or not r.stdout.strip():
            return []
        data = json.loads(r.stdout)
        return data if isinstance(data, list) else [data]
    except Exception:
        return []


def _wmic_list(query):
    """wmic /format:list -> dict (仅限 wmic 可用的系统)。"""
    try:
        r = subprocess.run(["wmic", query, "/format:list"], capture_output=True,
                           text=True, timeout=20, encoding="utf-8", errors="replace",
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if r.returncode != 0 or not r.stdout.strip():
            return {}
        d = {}
        for line in r.stdout.splitlines():
            if "=" in line:
                k, _, v = line.partition("=")
                d[k.strip()] = v.strip()
        return d
    except Exception:
        return {}


def _query_objects(query):
    """先 wmic 后 PowerShell, 返回 list[dict]。"""
    w = _wmic_list(query)
    if w:
        return [w]
    return _ps_json(query)


def detect_os():
    info = {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "arch": platform.architecture()[0],
    }
    if sys.platform == "win32":
        # platform.release() 在 Win11 上可能返回 "10", 用注册表判定真实版本
        try:
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                 r"SOFTWARE\Microsoft\Windows NT\CurrentVersion")
            try:
                prod = winreg.QueryValueEx(key, "ProductName")[0]
                cur = winreg.QueryValueEx(key, "DisplayVersion")[0]
                info["friendly"] = f"{prod} ({cur}, {info['arch']})"
            finally:
                winreg.CloseKey(key)
        except Exception:
            info["friendly"] = f"Windows {info['release']} ({info['arch']})"
    return info


def detect_cpu():
    """CPU 名称 + 物理/逻辑核心数。Windows 用注册表拿名称。"""
    name = platform.processor() or "Unknown CPU"
    try:
        if sys.platform == "win32":
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                 r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            val, _ = winreg.QueryValueEx(key, "ProcessorNameString")
            name = val.strip()
            winreg.CloseKey(key)
    except Exception:
        pass
    cores_phys = os.cpu_count() or 0
    cores_logical = cores_phys
    if sys.platform == "win32":
        objs = _query_objects("Win32_Processor")
        if objs:
            o = objs[0]
            try:
                cores_phys = int(o.get("NumberOfCores") or cores_phys)
                cores_logical = int(o.get("NumberOfLogicalProcessors") or cores_logical)
            except (TypeError, ValueError):
                pass
    return {"name": name, "cores_physical": cores_phys, "cores_logical": cores_logical}


def detect_ram():
    """总内存 / 可用内存 (GB)。Windows 用 GlobalMemoryStatusEx 拿真实总量。"""
    total = shutil_ram = 0
    try:
        shutil_ram = shutil_ram_bytes()
    except Exception:
        pass
    if sys.platform == "win32":
        try:
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong),
                            ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            m = MEMORYSTATUSEX()
            m.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
                total = m.ullTotalPhys
        except Exception:
            pass
    if not total:
        total = shutil_ram
    avail = 0
    if sys.platform == "win32":
        objs = _query_objects("Win32_OperatingSystem")
        if objs:
            o = objs[0]
            try:
                avail = int(o.get("FreePhysicalMemory") or 0) * 1024  # KB -> bytes
            except (TypeError, ValueError):
                avail = 0
    return {"total_gb": round(total / 1024**3, 1), "free_gb": round(avail / 1024**3, 1)}


def shutil_ram_bytes():
    """兜底: 某些环境下 shutil 没有直接 API, 用 os.sysconf (POSIX)。"""
    if hasattr(os, "sysconf"):
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        if pages > 0 and page_size > 0:
            return pages * page_size
    return 0


def detect_gpu():
    """GPU 检测:
    1) nvidia-smi (最准确, 给显存 + 驱动)
    2) Windows 注册表 / PowerShell WMI 兜底 (AMD/Intel/NVIDIA 通用)
    返回: [{"name", "vram_gb", "vendor", "driver"}]  (vram_gb 可能为 0 = 未知)
    """
    gpus = []
    # 1) NVIDIA
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if r.returncode == 0:
            for line in r.stdout.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 2:
                    vram = 0
                    try:
                        vram = round(int(parts[1]) / 1024, 1)  # MiB -> GB
                    except ValueError:
                        pass
                    gpus.append({"name": parts[0], "vram_gb": vram,
                                 "vendor": "nvidia",
                                 "driver": parts[2] if len(parts) > 2 else ""})
    except Exception:
        pass
    if gpus:
        return gpus
    # 2) WMI 兜底
    try:
        if sys.platform == "win32":
            for o in _query_objects("Win32_VideoController"):
                n = o.get("Name") or ""
                vram = 0
                try:
                    vram = round(int(o.get("AdapterRAM") or 0) / 1024**3, 1)
                except (TypeError, ValueError):
                    vram = 0
                vendor = "unknown"
                nl = n.lower()
                if "nvidia" in nl or "geforce" in nl or "quadro" in nl:
                    vendor = "nvidia"
                elif "amd" in nl or "radeon" in nl:
                    vendor = "amd"
                elif "intel" in nl or "arc" in nl:
                    vendor = "intel"
                gpus.append({"name": n, "vram_gb": vram, "vendor": vendor, "driver": ""})
    except Exception:
        pass
    return gpus


def detect_disks():
    """所有盘符 + 剩余空间 GB。"""
    disks = []
    if sys.platform == "win32":
        for letter in "CDEFGHIJKLMNOPQRSTUVWXYZ":
            root = letter + ":\\"
            if os.path.exists(root):
                try:
                    u = shutil.disk_usage(root)
                    disks.append({"mount": root, "free_gb": round(u.free / 1024**3, 1),
                                  "total_gb": round(u.total / 1024**3, 1)})
                except Exception:
                    continue
    else:
        try:
            u = shutil.disk_usage("/")
            disks.append({"mount": "/", "free_gb": round(u.free / 1024**3, 1),
                          "total_gb": round(u.total / 1024**3, 1)})
        except Exception:
            pass
    return disks


def detect_all():
    """汇总检测。返回 dict, 供安装器和 UI 使用。"""
    return {
        "os": detect_os(),
        "cpu": detect_cpu(),
        "ram": detect_ram(),
        "gpus": detect_gpu(),
        "disks": detect_disks(),
        "python": sys.version.split()[0],
    }


def has_nvidia_gpu(gpus=None):
    gpus = gpus if gpus is not None else detect_gpu()
    return any(g["vendor"] == "nvidia" for g in gpus)


def best_disk_free_gb(disks=None):
    disks = disks if disks is not None else detect_disks()
    return max((d["free_gb"] for d in disks), default=0)


if __name__ == "__main__":
    print(json.dumps(detect_all(), ensure_ascii=False, indent=2))
