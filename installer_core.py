# -*- coding: utf-8 -*-
"""安装核心 — 下载并部署 llama.cpp + GGUF 模型, 生成启动脚本。

特性:
  - llama.cpp 从 GitHub Releases 下载 (CUDA / CPU / Vulkan 版自动选择)
  - GGUF 从 HuggingFace 下载, 国内自动走 hf-mirror.com, 失败回退官方源
  - 断点续传 (HTTP Range), 带进度条, 失败重试
  - 生成 start-server.cmd / chat.cmd 等启动脚本
"""
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
import urllib.error
import zipfile
from pathlib import Path

# ---------------------------------------------------------------- 常量

LLAMA_CPP_REPO = "ggml-org/llama.cpp"
LLAMA_CPP_TAG = "b10424"          # 2026-08-14 核实的最新版

GITHUB_MIRRORS = [
    "https://github.com",          # 直连
    "https://ghfast.top/https://github.com",
    "https://gh-proxy.com/https://github.com",
    "https://mirror.ghproxy.com/https://github.com",
]
HF_MIRRORS = [
    "https://hf-mirror.com",       # 国内主镜像 (快)
    "https://huggingface.co",      # 官方 (回退)
]
# 魔搭 (ModelScope, 阿里): 国内满速 (~7MB/s), Qwen 官方 GGUF 仓库。
# 仅部分模型有 Q4_K_M (4B/8B/14B/32B 有; 1.7B 只有 Q8_0; 35B-A3B 没有)。
MODELSCOPE_BASE = "https://modelscope.cn"
MODELSCOPE_REPOS = {
    "qwen3-4b":  "Qwen/Qwen3-4B-GGUF",
    "qwen3-8b":  "Qwen/Qwen3-8B-GGUF",
    "qwen3-14b": "Qwen/Qwen3-14B-GGUF",
    "qwen3-32b": "Qwen/Qwen3-32B-GGUF",
}

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) LocalAI-Installer/1.0"


# ---------------------------------------------------------------- 下载

def _request(url, headers=None, timeout=30):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, **(headers or {})})
    return urllib.request.urlopen(req, timeout=timeout)


def probe_url(url, timeout=15):
    """HEAD 探测 URL 是否可用, 返回 (ok, content_length)"""
    try:
        req = urllib.request.Request(url, method="HEAD",
                                     headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return True, int(r.headers.get("Content-Length") or 0)
    except Exception:
        return False, 0


def download(url, dest, progress_cb=None, retries=5):
    """下载文件, 断点续传 + 重试。progress_cb(downloaded, total)。
    连接中断/文件不完整会被检测并重试, 绝不把残缺文件当成功。
    SSL EOF 等间歇性断连会退避重试多次; .part 断点文件始终保留, 可跨进程续传。
    返回最终文件大小 (字节); 失败抛异常。"""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    for attempt in range(1, retries + 1):
        resume = tmp.stat().st_size if tmp.exists() else 0
        headers = {"User-Agent": USER_AGENT}
        if resume:
            headers["Range"] = f"bytes={resume}-"
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=60) as resp:
                total = int(resp.headers.get("Content-Length") or 0) + resume
                mode = "ab" if resume and resp.status == 206 else "wb"
                if mode == "wb":
                    resume = 0
                    total = int(resp.headers.get("Content-Length") or 0)
                with open(tmp, mode) as f:
                    while True:
                        chunk = resp.read(1024 * 256)
                        if not chunk:
                            break
                        f.write(chunk)
                        if progress_cb:
                            progress_cb(f.tell(), total)
            size = tmp.stat().st_size
            # 完整性校验: 已知总大小时, 不允许缺字节
            if total > 0 and size < total:
                raise RuntimeError(f"连接中断: 仅收到 {size}/{total} 字节 (尝试 {attempt}/{retries})")
            if size > 0:
                os.replace(tmp, dest)
                return dest.stat().st_size
        except RuntimeError as e:
            if attempt >= retries:
                raise
            time.sleep(3 * attempt)
        except Exception as e:
            if attempt >= retries:
                raise RuntimeError(f"下载失败 (第{attempt}次): {url}\n  {e}")
            # 间歇性断连 (SSL EOF 等): 指数退避后从断点续传
            time.sleep(3 * attempt)
    raise RuntimeError(f"下载失败: {url}")


def try_mirrors(urls, dest, progress_cb=None, retries=2, desc=""):
    """依次尝试多个镜像, 返回成功下载的 URL; 全失败抛异常。
    关键: 失败时保留 .part 断点文件, 下一个镜像从断点续传, 不浪费已下字节。"""
    dest = Path(dest)
    last_err = None
    for u in urls:
        try:
            if progress_cb:
                progress_cb(-1, -1, f"尝试: {u}")
            download(u, dest, progress_cb, retries)
            return u
        except Exception as e:
            last_err = e
            # 只删残缺的最终文件; .part 断点文件保留供续传
            if dest.exists():
                dest.unlink()
            continue
    raise RuntimeError(f"{desc or '下载'}失败: {last_err}")


# ---------------------------------------------------------------- llama.cpp

# RTX 50 系 (Blackwell, sm_120) 需要 CUDA >= 12.8。llama.cpp 的 CUDA 13.3 构建
# 需要配套 cudart 13.3 runtime DLL 且驱动 >= 对应版本 (591.x 只到 CUDA 13.1);
# 驱动不够新时, 50 系应回退 Vulkan 构建 (无 runtime 依赖, 速度接近 CUDA)。
BLACKWELL_MARKERS = ("rtx 50", "rtx 5060", "rtx 5070", "rtx 5080", "rtx 5090",
                     "rtx 5060ti", "rtx 5070ti", "rtx 5080ti", "rtx 5090d",
                     "blackwell", "b200", "b100", "gb200", "gb100", "rtx pro 50")

# CUDA runtime minor -> 所需最小驱动版本 (粗估: CUDA 13.x 每 minor 对应驱动阶梯)
# 驱动版本 >= 表值时 CUDA 13.3 构建可用, 否则回退 Vulkan
CUDA_13_3_MIN_DRIVER = 592.0


def is_blackwell(gpu_name=""):
    nl = (gpu_name or "").lower()
    return any(m in nl for m in BLACKWELL_MARKERS)


def driver_supports_cuda_13_3(driver=""):
    """驱动版本足够新则可用 CUDA 13.3 构建, 否则回退 Vulkan。"""
    try:
        return float(driver) >= CUDA_13_3_MIN_DRIVER
    except (TypeError, ValueError):
        return False


def llama_cpp_asset_name(engine, gpu_name=""):
    """根据推理引擎返回 release 资产名 (win-x64)。"""
    if engine == "cuda":
        cuda_ver = "13.3" if is_blackwell(gpu_name) else "12.4"
        return f"llama-{LLAMA_CPP_TAG}-bin-win-cuda-{cuda_ver}-x64.zip"
    if engine == "vulkan":
        return f"llama-{LLAMA_CPP_TAG}-bin-win-vulkan-x64.zip"
    return f"llama-{LLAMA_CPP_TAG}-bin-win-cpu-x64.zip"


def llama_cpp_urls(engine, gpu_name=""):
    asset = llama_cpp_asset_name(engine, gpu_name)
    return [f"{m}/{LLAMA_CPP_REPO}/releases/download/{LLAMA_CPP_TAG}/{asset}"
            for m in GITHUB_MIRRORS]


def install_llama_cpp(install_dir, engine, progress_cb=None, gpu_name=""):
    """下载并解压 llama.cpp 到 install_dir/llama.cpp/, 返回 llama-server.exe 路径。
    已安装 (存在 llama-server.exe) 则跳过下载。"""
    dest_dir = Path(install_dir) / "llama.cpp"
    existing = find_exe(dest_dir, "llama-server") if dest_dir.exists() else None
    if existing:
        print(f"llama.cpp 已安装, 跳过下载: {existing}")
        return existing
    zip_path = Path(tempfile.gettempdir()) / f"llama-{engine}-{LLAMA_CPP_TAG}.zip"

    print(f"[1/3] 下载 llama.cpp ({engine} 版, ~200MB)...")
    try_mirrors(llama_cpp_urls(engine), zip_path, progress_cb, desc="llama.cpp")

    print("[2/3] 解压中...")
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(dest_dir)
    zip_path.unlink(missing_ok=True)

    server = find_exe(dest_dir, "llama-server")
    if not server:
        raise RuntimeError("解压后未找到 llama-server.exe")
    print(f"[3/3] llama.cpp 就绪: {server}")
    return str(server)


def find_exe(root, name):
    """在解压目录里递归找 exe (release 包可能带一层子目录)。"""
    for p in Path(root).rglob(name + ".exe"):
        return str(p)
    return None


# ---------------------------------------------------------------- 模型

def model_urls(repo, filename, model_id=None):
    """模型下载 URL 列表: 优先魔搭 (国内满速), 再 hf-mirror, 最后 HF 官方。"""
    urls = []
    ms_repo = MODELSCOPE_REPOS.get(model_id or "")
    if ms_repo:
        urls.append(f"{MODELSCOPE_BASE}/models/{ms_repo}/resolve/master/{filename}")
    urls += [f"{m}/{repo}/resolve/main/{filename}" for m in HF_MIRRORS]
    return urls


def expected_model_size(repo, filename, model_id=None):
    """HEAD 请求获取模型真实大小 (用于完整性校验)。"""
    for base in ([MODELSCOPE_BASE] if model_id in MODELSCOPE_REPOS else []) + HF_MIRRORS:
        try:
            url = (f"{base}/models/{MODELSCOPE_REPOS[model_id]}/resolve/master/{filename}"
                   if base == MODELSCOPE_BASE
                   else f"{base}/{repo}/resolve/main/{filename}")
            ok, size = probe_url(url, timeout=20)
            if ok and size > 0:
                return size
        except Exception:
            continue
    return 0


def install_model(install_dir, model, progress_cb=None):
    """下载 GGUF 到 install_dir/models/, 返回模型文件路径。
    已存在但大小不对 (下载中断残留) 会重新下载。"""
    models_dir = Path(install_dir) / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    dest = models_dir / model["file"]

    expected = expected_model_size(model["repo"], model["file"], model.get("id"))
    if dest.exists() and dest.stat().st_size > 10**6:
        if expected and abs(dest.stat().st_size - expected) > expected * 0.01:
            print(f"模型文件不完整 ({dest.stat().st_size}/{expected} 字节), 重新下载")
            dest.unlink()
            dest.with_suffix(dest.suffix + ".part").unlink(missing_ok=True)
        else:
            print(f"模型已存在, 跳过下载: {dest.name}")
            return str(dest)

    print(f"[下载模型] {model['name']} ({model['size_gb']}GB)")
    try_mirrors(model_urls(model["repo"], model["file"], model.get("id")), dest, progress_cb,
                desc=f"模型 {model['file']}")
    # 下载后二次校验: HEAD 大小与实际文件可能有微小差异 (魔搭差 1KB),
    # 允许 1% 容差; download() 内部已用 GET Content-Length 严格校验过完整性
    if expected and dest.stat().st_size < expected * 0.99:
        raise RuntimeError(f"模型下载不完整: {dest.stat().st_size}/{expected} 字节")
    print(f"模型就绪: {dest}")
    return str(dest)


# ---------------------------------------------------------------- 启动脚本

def write_launch_scripts(install_dir, model_path, engine, ngl, ctx, server_port=8080):
    """生成 start-server.cmd / start-chat.cmd / stop-server.cmd"""
    install_dir = Path(install_dir)
    server_exe = Path(find_exe(install_dir / "llama.cpp", "llama-server") or "")
    if not server_exe.exists():
        raise RuntimeError("未找到 llama-server.exe, 安装不完整")
    scripts = install_dir
    scripts.mkdir(parents=True, exist_ok=True)

    # ---- start-server.cmd
    start_server = scripts / "start-server.cmd"
    # Windows cmd 转义: % 写成 %%
    start_server.write_text(
        f"""@echo off
chcp 65001 >nul
title Local AI - llama-server (port {server_port})
echo 正在启动本地大模型服务...
"{server_exe}" -m "{model_path}" --port {server_port} -c {ctx} -ngl {ngl} --jinja --alias "local-ai"
echo.
echo 服务已退出 (按任意键关闭)
pause >nul
""", encoding="utf-8")

    # ---- start-chat.cmd (零依赖: 用 llama.cpp 自带的 llama-cli)
    cli_exe = Path(find_exe(install_dir / "llama.cpp", "llama-cli") or "")
    start_chat = scripts / "start-chat.cmd"
    if cli_exe.exists():
        start_chat.write_text(
            f"""@echo off
chcp 65001 >nul
title Local AI - 聊天
echo 正在连接本地服务... (若服务未启动, 请先运行 start-server.cmd)
"{cli_exe}" -m "{model_path}" -c {ctx} -ngl {ngl} --jinja --conversation
pause
""", encoding="utf-8")
    else:
        # 回退: 用 mini_agent; 优先用打包好的 mini-agent.exe (零依赖),
        # 没有则复制 mini_agent.py 并需要 Python
        agent_exe = install_dir / "mini-agent.exe"
        src_exe = Path(__file__).resolve().parent / "mini-agent.exe"
        try:
            if src_exe.exists():
                import shutil as _sh
                _sh.copy2(src_exe, agent_exe)
        except Exception:
            pass
        if agent_exe.exists():
            start_chat.write_text(
                f"""@echo off
chcp 65001 >nul
title Local AI - 聊天
echo 正在连接本地服务... (若服务未启动, 请先运行 start-server.cmd)
"{agent_exe}" --url http://127.0.0.1:{server_port}/v1 --model local-ai
pause
""", encoding="utf-8")
        else:
            agent_dst = install_dir / "mini_agent.py"
            src = Path(__file__).resolve().parent / "mini_agent.py"
            try:
                if src.exists():
                    import shutil as _sh
                    _sh.copy2(src, agent_dst)
            except Exception:
                pass
            agent_script = agent_dst if agent_dst.exists() else src
            start_chat.write_text(
                f"""@echo off
chcp 65001 >nul
title Local AI - 聊天
python "{agent_script}" --url http://127.0.0.1:{server_port}/v1 --model local-ai
pause
""", encoding="utf-8")

    # ---- stop-server.cmd
    stop_server = scripts / "stop-server.cmd"
    stop_server.write_text(
        f"""@echo off
taskkill /f /im llama-server.exe >nul 2>&1
echo llama-server 已停止
timeout /t 1 >nul
""", encoding="utf-8")

    return str(start_server), str(start_chat)


# ---------------------------------------------------------------- 服务控制

def find_free_port(start=8080, tries=20):
    """从 start 起找第一个空闲端口 (处理 ollama 等占用 8080 的情况)。"""
    import socket
    for port in range(start, start + tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    return start


def wait_server(url, timeout=120):
    """等待 OpenAI 兼容服务就绪。"""
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with _request(f"{url}/models", timeout=5) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(2)
    return False


def start_server_background(server_exe, model_path, port=8080, ctx=8192, ngl=999):
    """后台启动 llama-server, 返回 Popen 对象。"""
    cmd = [server_exe, "-m", model_path, "--port", str(port),
           "-c", str(ctx), "-ngl", str(ngl), "--jinja", "--alias", "local-ai"]
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))


def test_chat(url, prompt="你好, 请用一句话自我介绍。", timeout=120):
    """调用 /v1/chat/completions 验证模型可用, 返回回复文本。"""
    body = json.dumps({
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 128,
        "temperature": 0.7,
    }).encode()
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            req = urllib.request.Request(
                f"{url}/chat/completions", data=body,
                headers={"Content-Type": "application/json", "User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=60) as r:
                data = json.loads(r.read())
            return data["choices"][0]["message"]["content"]
        except Exception:
            time.sleep(3)
    return None
