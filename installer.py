# -*- coding: utf-8 -*-
"""Local AI Installer — 自动检测电脑并安装适配的本地大模型 + 配套 Agent。

用法:
    python installer.py                # 交互式向导
    python installer.py --yes --dir D:\\local-ai   # 全自动 (用推荐配置)
    python installer.py --list-models  # 只列推荐, 不安装
    python installer.py --engine cpu   # 强制 CPU 版 llama.cpp

流程: 检测硬件 -> 推荐模型 -> 下载 llama.cpp -> 下载 GGUF -> 启动验证 -> 配置 Hermes。
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hardware
import recommender
import installer_core as core
import hermes_setup

VERSION = "1.0.0"
GREEN, YELLOW, RED, CYAN, RESET = "\033[32m", "\033[33m", "\033[31m", "\033[36m", "\033[0m"


def cprint(color, msg):
    print(f"{color}{msg}{RESET}")


def ensure_utf8():
    """Windows 控制台可能默认 GBK, 强制 UTF-8 输出。"""
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
        os.system("chcp 65001 >nul 2>&1")


def ask(question, default=None):
    """交互提问, 返回用户输入; 空输入返回 default。"""
    if default is not None:
        q = f"{question} [{default}] "
    else:
        q = f"{question} "
    try:
        ans = input(q).strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return default
    return ans or default


def ask_yes(question, default=True):
    d = "Y/n" if default else "y/N"
    ans = ask(question, d).lower()
    if ans in ("", d.lower()):
        return default
    return ans.startswith("y")


def show_hardware(hw):
    print("\n" + "=" * 62)
    cprint(CYAN, "   电脑硬件检测结果")
    print("=" * 62)
    o = hw["os"]
    print(f"  操作系统 : {o.get('friendly') or (o['system'] + ' ' + o.get('release', ''))}")
    c = hw["cpu"]
    print(f"  CPU      : {c['name']}")
    print(f"            物理 {c['cores_physical']} 核 / 逻辑 {c['cores_logical']} 线程")
    r = hw["ram"]
    print(f"  内存     : {r['total_gb']} GB (可用 {r['free_gb']} GB)")
    for g in hw["gpus"]:
        vram = f"{g['vram_gb']} GB" if g["vram_gb"] else "?"
        print(f"  显卡     : {g['name']} ({vram} 显存, {g['vendor']})")
    if not hw["gpus"]:
        print("  显卡     : 未检测到独立显卡 (纯 CPU 推理)")
    best = max((d["free_gb"] for d in hw["disks"]), default=0)
    print(f"  磁盘     : 最大剩余空间 {best} GB")
    print("=" * 62 + "\n")


def progress_bar(desc=""):
    """返回进度回调闭包。"""
    last = [0]

    def cb(done, total, note=None):
        if done is None:
            return
        if done < 0:
            if note:
                print(f"  {note}")
            return
        if total <= 0:
            return
        pct = done * 100 // total
        if pct >= last[0] + 2 or pct == 100:
            last[0] = pct
            bar = "#" * (pct // 4) + "-" * (25 - pct // 4)
            mb = done / 1024**2
            print(f"\r  [{bar}] {pct:3d}%  {mb:6.1f} MB / {total/1024**2:6.1f} MB",
                  end="", flush=True)
            if pct == 100:
                print()
    return cb


def list_models(hw):
    recos = recommender.recommend(hw)
    print("\n根据你的硬件, 推荐模型 (按适配度排序):")
    for i, r in enumerate(recos, 1):
        m = r["model"]
        fit = {"exact": "最适配", "tight": "可用", "loose": "保底"}[r["fit"]]
        print(f"  [{i}] {m['name']}  ({m['size_gb']}GB, {fit}, {r['engine']})")
        print(f"      {m['desc']}")
    return recos


def choose_model(recos):
    if not recos:
        cprint(RED, "没有找到适配的模型! 请检查硬件检测结果。")
        sys.exit(1)
    if len(recos) == 1:
        m = recos[0]["model"]
        print(f"\n自动选择: {m['name']} ({m['size_gb']}GB)")
        return recos[0]
    print("\n请选择要安装的模型:")
    for i, r in enumerate(recos, 1):
        m = r["model"]
        print(f"  [{i}] {m['name']}  ({m['size_gb']}GB) — {m['desc']}")
    choice = ask("输入序号", "1")
    try:
        idx = int(choice) - 1
        return recos[idx]
    except (ValueError, IndexError):
        cprint(YELLOW, "输入无效, 使用默认推荐 (1)。")
        return recos[0]


def choose_install_dir(hw):
    default = str(Path(os.environ.get("LOCALAPPDATA", Path.home())) / "local-ai")
    print(f"\n默认安装目录: {default}")
    d = ask("安装目录 (留空用默认)")
    d = d or default
    d = os.path.abspath(d)
    # 检查磁盘空间
    try:
        free = shutil.disk_usage(d[:3] if len(d) > 2 else d).free / 1024**3
        if free < 20:
            cprint(YELLOW, f"警告: {d[:3]} 剩余空间仅 {free:.1f}GB, 可能不够。")
    except Exception:
        pass
    return d


def print_banner():
    print()
    print("  ╔══════════════════════════════════════════════════╗")
    print("  ║       Local AI Installer  v" + VERSION.ljust(20) + "     ║")
    print("  ║   自动检测电脑 · 安装本地大模型 · 一键聊天       ║")
    print("  ╚══════════════════════════════════════════════════╝")
    print("  本工具将: 检测硬件 → 推荐模型 → 下载 llama.cpp 和模型")
    print("            → 生成启动脚本 → (可选) 配置 Hermes Agent\n")


def main():
    ensure_utf8()
    ap = argparse.ArgumentParser(description="Local AI Installer")
    ap.add_argument("--yes", "-y", action="store_true", help="全自动 (默认推荐配置)")
    ap.add_argument("--dir", help="安装目录")
    ap.add_argument("--engine", choices=["auto", "cuda", "vulkan", "cpu"], default="auto",
                    help="推理引擎 (默认按显卡自动选)")
    ap.add_argument("--list-models", action="store_true", help="只列推荐模型")
    ap.add_argument("--model", help="指定模型 id (如 qwen3-8b, qwen3-1.7b), 默认按硬件推荐")
    ap.add_argument("--no-hermes", action="store_true", help="不配置 Hermes")
    ap.add_argument("--skip-verify", action="store_true", help="装完不启动验证")
    ap.add_argument("--port", type=int, default=8080, help="llama-server 端口")
    args = ap.parse_args()

    print_banner()

    # 1. 检测硬件
    print("正在检测硬件...")
    hw = hardware.detect_all()
    show_hardware(hw)

    recos = recommender.recommend(hw)
    if args.list_models:
        list_models(hw)
        sys.exit(0)

    # 2. 选择模型
    chosen = recos[0]
    if args.model:
        found = [r for r in recos if r["model"]["id"] == args.model]
        if found:
            chosen = found[0]
        else:
            # 允许指定推荐外的模型 (用户明确要求)
            m = next((x for x in recommender.MODELS if x["id"] == args.model), None)
            if m:
                chosen = {"model": m, "engine": chosen["engine"], "ngl": chosen["ngl"],
                          "ctx": 8192, "reason": "用户指定", "fit": "exact"}
            else:
                cprint(RED, f"未知模型 id: {args.model} (可用: {', '.join(x['id'] for x in recommender.MODELS)})")
                sys.exit(1)
    elif not args.yes:
        list_models(hw)
        chosen = choose_model(recos)

    model = chosen["model"]
    engine = args.engine if args.engine != "auto" else chosen["engine"]

    # 3. 安装目录
    install_dir = args.dir or (None if args.yes else None)
    if not install_dir:
        install_dir = choose_install_dir(hw)

    # 4. 确认
    total_need = model["size_gb"] + 0.6  # 模型 + llama.cpp
    # 端口冲突检测: 8080 可能被 ollama 等占用
    if args.port == 8080:
        actual_port = core.find_free_port(8080)
        if actual_port != 8080:
            cprint(YELLOW, f"注意: 端口 8080 被其他程序占用, 自动改用端口 {actual_port}")
    else:
        actual_port = args.port
    print(f"\n即将安装:")
    print(f"  - 推理引擎 : llama.cpp ({engine} 版)")
    print(f"  - 模型     : {model['name']} ({model['size_gb']} GB)")
    print(f"  - 安装目录 : {install_dir}")
    print(f"  - 服务端口 : {actual_port}")
    print(f"  - 预计占用 : ~{total_need:.1f} GB")
    if not args.yes:
        if not ask_yes("\n开始安装?", True):
            print("已取消。")
            sys.exit(0)

    # 5. 执行安装
    t0 = time.time()
    gpu_name = hw["gpus"][0]["name"] if hw.get("gpus") else ""
    try:
        server_exe = core.install_llama_cpp(install_dir, engine, progress_bar(), gpu_name)
        model_path = core.install_model(install_dir, model, progress_bar())
    except RuntimeError as e:
        cprint(RED, f"\n安装失败: {e}")
        sys.exit(1)

    # 6. 生成启动脚本
    start_server, start_chat = core.write_launch_scripts(
        install_dir, model_path, engine, chosen["ngl"], chosen["ctx"], actual_port)
    print(f"\n启动脚本已生成:")
    print(f"  - 启动服务 : {start_server}")
    print(f"  - 聊天     : {start_chat}")

    # 7. 启动验证 (后台)
    if not args.skip_verify:
        print(f"\n正在启动服务并验证 (端口 {actual_port}, 首次加载模型需 10-60 秒)...")
        proc = core.start_server_background(server_exe, model_path,
                                            actual_port, chosen["ctx"], chosen["ngl"])
        try:
            if core.wait_server(f"http://127.0.0.1:{actual_port}/v1", timeout=180):
                print("服务已就绪 ✓")
                reply = core.test_chat(f"http://127.0.0.1:{actual_port}/v1",
                                       "你好, 请简单回复我")
                if reply:
                    print(f"验证对话: {reply[:120]}")
                else:
                    cprint(YELLOW, "验证对话无回复 (模型可能仍在加载, 稍后可用 start-chat.cmd 手动验证)")
            else:
                cprint(YELLOW, "服务启动超时, 可稍后手动运行 start-server.cmd。")
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except Exception:
                proc.kill()
            print("(验证完成, 服务已停止; 日常使用请双击 start-server.cmd)")

    # 8. Hermes 集成
    if not args.no_hermes:
        print("\n" + "-" * 62)
        cprint(CYAN, "Hermes Agent 集成")
        print("-" * 62)
        if hermes_setup.hermes_installed():
            print(f"检测到 Hermes: {hermes_setup.hermes_exe()}")
            if args.yes or ask_yes("配置 Hermes 使用本地模型 (独立 profile, 不影响现有配置)?", True):
                status, msg = hermes_setup.setup_hermes(
                    f"http://127.0.0.1:{actual_port}/v1", "local-ai")
                cprint(GREEN if status == "ok" else YELLOW, f"  [{status}] {msg}")
        else:
            cprint(YELLOW, "未检测到 Hermes Agent。")
            print("可稍后用 PowerShell 安装 (装完重跑本程序即可自动配置):")
            print(f"  {hermes_setup.INSTALL_PS1}")

    # 9. 完成
    elapsed = time.time() - t0
    print("\n" + "=" * 62)
    cprint(GREEN, "  ✅ 安装完成!")
    print("=" * 62)
    print(f"  使用方式:")
    print(f"    1. 双击 {start_server}   启动本地大模型服务")
    print(f"    2. 双击 {start_chat}     开始聊天")
    print(f"  耗时 {elapsed:.0f} 秒。")
    print("=" * 62 + "\n")


if __name__ == "__main__":
    main()
