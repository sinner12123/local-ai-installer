# -*- coding: utf-8 -*-
"""Hermes Agent 集成 — 检测/安装 Hermes, 配置其使用本地大模型。

安全策略: 绝不覆盖用户现有配置。
  - 若 Hermes 未安装: 提示可选安装 (PowerShell 一行命令)
  - 若已安装: 创建独立 profile "local", 在其中配置 custom provider
    指向本地 llama-server, 用户的默认 profile 完全不受影响。
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

INSTALL_PS1 = "iex (irm https://hermes-agent.nousresearch.com/install.ps1)"
PROFILE_NAME = "local"


def hermes_exe():
    """返回 hermes 可执行文件路径, 找不到返回 None。"""
    exe = shutil.which("hermes")
    if exe:
        return exe
    # 常见 Windows 安装位置
    for cand in (
        Path(os.environ.get("LOCALAPPDATA", "")) / "hermes" / "bin" / "hermes.exe",
        Path.home() / ".local" / "bin" / "hermes",
        Path.home() / ".hermes" / "bin" / "hermes.exe",
    ):
        if cand.exists():
            return str(cand)
    return None


def hermes_installed():
    return hermes_exe() is not None


def run(args, timeout=120, capture=True):
    """运行 hermes 子命令。"""
    exe = hermes_exe()
    if not exe:
        raise RuntimeError("未找到 hermes 可执行文件")
    cmd = [exe] + args
    try:
        r = subprocess.run(cmd, capture_output=capture, text=True, timeout=timeout,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        return r
    except Exception as e:
        raise RuntimeError(f"hermes 命令失败: {' '.join(cmd)}\n  {e}")


def profile_exists():
    r = run(["profile", "list"])
    return PROFILE_NAME in (r.stdout or "") + (r.stderr or "")


def create_profile():
    """创建独立 profile 'local' (仅当不存在)。"""
    if profile_exists():
        return
    r = run(["profile", "create", PROFILE_NAME])
    if r.returncode != 0:
        raise RuntimeError(f"创建 profile 失败: {r.stderr or r.stdout}")


def configure_local_model(server_url="http://127.0.0.1:8080/v1", model_name="local-ai"):
    """在 profile 'local' 中配置 custom provider -> 本地 llama-server。"""
    create_profile()
    base = ["--profile", PROFILE_NAME, "config", "set"]
    steps = [
        (*base, "model.provider", "custom"),
        (*base, "model.base_url", server_url),
        (*base, "model.api_key", "local-no-key"),
        (*base, "model.model", model_name),
    ]
    for s in steps:
        r = run(list(s))
        if r.returncode != 0:
            raise RuntimeError(f"配置失败: {' '.join(s)}\n  {r.stderr or r.stdout}")
    return True


def setup_hermes(server_url="http://127.0.0.1:8080/v1", model_name="local-ai",
                 auto_install=False):
    """完整集成: 检测 -> (可选安装) -> 配置独立 profile。
    返回 (状态, 说明)。"""
    if not hermes_installed():
        if not auto_install:
            return ("not_installed",
                    f"Hermes 未安装。可用 PowerShell 一键安装:\n  {INSTALL_PS1}\n"
                    "安装后重新运行本安装器即可自动配置。")
        # 尝试自动安装 (PowerShell)
        print("检测到 Hermes 未安装, 正在自动安装 (需几分钟)...")
        ps = "Set-ExecutionPolicy Bypass -Scope Process -Force; " + INSTALL_PS1
        r = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                           timeout=900)
        if r.returncode != 0 or not hermes_installed():
            return ("install_failed",
                    f"Hermes 自动安装失败, 请手动运行:\n  {INSTALL_PS1}")
    try:
        configure_local_model(server_url, model_name)
        return ("ok", f"Hermes 已配置本地模型 (profile: {PROFILE_NAME})。\n"
                      f"使用方式: hermes --profile {PROFILE_NAME}")
    except Exception as e:
        return ("config_failed", f"Hermes 配置失败: {e}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080/v1")
    ap.add_argument("--model", default="local-ai")
    ap.add_argument("--auto-install", action="store_true")
    a = ap.parse_args()
    status, msg = setup_hermes(a.url, a.model, a.auto_install)
    print(f"[{status}] {msg}")
