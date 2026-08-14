# Local AI Installer

自动检测电脑硬件, 安装最适配的本地大模型 + 配套 Agent 的一键安装器。

## 功能

- **自动硬件检测** — CPU / 内存 / 显卡 (NVIDIA/AMD/Intel) / 磁盘空间 / 操作系统
- **智能模型推荐** — 根据显存和内存推荐最适配的 Qwen3 GGUF 模型 (1.7B ~ 32B)
- **自动选推理引擎** — NVIDIA 显卡用 CUDA 版, AMD/Intel 用 Vulkan 版, 无独显用 CPU 版
  (RTX 50 系 Blackwell 自动用 CUDA 13.3 构建)
- **国内网络友好** — 模型走 hf-mirror.com, 失败自动回退 huggingface.co;
  llama.cpp 走 GitHub 直连 + 多个加速镜像回退; 全部支持断点续传
- **自带轻量 Agent** — 装完双击 start-chat.cmd 即可聊天 (mini_agent, 零依赖)
- **Hermes Agent 集成** — 自动检测已安装的 Hermes, 创建独立 profile 指向本地模型,
  绝不触碰用户现有配置
- **启动脚本** — start-server.cmd (起服务) / start-chat.cmd (聊天) / stop-server.cmd (停)

## 快速开始

```bash
# 交互式向导
python installer.py

# 全自动 (推荐配置, 安装到默认目录)
python installer.py --yes

# 全自动安装到指定目录
python installer.py --yes --dir D:\local-ai

# 只看推荐不安装
python installer.py --list-models
```

## 使用 (安装完成后)

```
1. 双击 start-server.cmd   启动本地大模型服务 (llama-server, 端口 8080)
2. 双击 start-chat.cmd     打开聊天窗口 (llama-cli, 零依赖)
3. (可选) Hermes 用户:  hermes --profile local
```

聊天有两种方式:
- **start-chat.cmd** — llama.cpp 自带 llama-cli, 无需 Python, 直接多轮对话
- **mini_agent.py** — 需要 Python, 连接常驻服务 (流式输出, 支持上下文管理):
  `python mini_agent.py --url http://127.0.0.1:8080/v1 --model local-ai`

## 打包成 exe

```bash
pip install pyinstaller
build.bat
```

生成 `dist/LocalAI-Installer.exe` (安装器) 和 `dist/mini-agent.exe` (自带聊天 Agent),
双击即用 (无需安装 Python)。

## 项目结构

```
installer.py        主安装器 (向导)
hardware.py         硬件检测 (纯 stdlib)
recommender.py      模型推荐引擎
installer_core.py   下载/部署 llama.cpp + 模型 + 启动脚本
mini_agent.py       自带轻量聊天 Agent (纯 stdlib)
hermes_setup.py     Hermes 集成 (独立 profile)
```

## 技术说明

- llama.cpp: 官方 GitHub Releases (b10424)
  - NVIDIA 老显卡: CUDA 12.4 构建
  - RTX 50 系 (Blackwell): 优先 CUDA 13.3 (需新驱动), 驱动不够新自动回退 Vulkan 构建
  - AMD/Intel 独显: Vulkan; 无独显: CPU
  (注意: CUDA 13.3 构建需要配套 cudart 13.3 runtime, 且驱动版本足够新)
- 模型: unsloth / Qwen 官方 GGUF (Q4_K_M 量化)
  - 下载源优先级: 魔搭 ModelScope (国内满速 ~7MB/s) → hf-mirror → HuggingFace 官方
- 下载: 断点续传 + 完整性校验 (绝不把残缺文件当成功); 换镜像保留 .part 续传
- 端口: 自动检测 8080 占用 (ollama 等), 冲突时自动换空闲端口
- Qwen3 思考模式: 本地聊天默认关闭 (`--reasoning off`), 否则回复被思考截断且变慢
- 显存不足提示: 若显卡显存被其他程序占用 (浏览器/Electron 应用等),
  GPU 加速可能不可用, 会自动降级 CPU 推理
- 模型下载后默认在 `<安装目录>/models/`, 引擎在 `<安装目录>/llama.cpp/`
- 服务: llama-server OpenAI 兼容 API, 默认 `http://127.0.0.1:8080/v1`
- mini_agent 通过 OpenAI 兼容接口对话, 支持多轮上下文和流式输出

## License

MIT
