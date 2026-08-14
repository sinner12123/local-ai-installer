# -*- coding: utf-8 -*-
"""mini_agent — 安装器自带的轻量本地 AI 助手。

零第三方依赖 (纯标准库), 通过 OpenAI 兼容 API 连接本地 llama-server:
    python mini_agent.py                # 默认连 http://127.0.0.1:8080/v1
    python mini_agent.py --url http://127.0.0.1:8080/v1 --model local-ai
    python mini_agent.py -q "你好"       # 单次提问 (脚本用)

功能: 多轮上下文对话, 流式输出, /new 清空, /exit 退出, 自动检测服务。
"""
import argparse
import json
import sys
import urllib.request


def build_parser():
    p = argparse.ArgumentParser(description="Local AI 轻量聊天助手")
    p.add_argument("--url", default="http://127.0.0.1:8080/v1",
                   help="OpenAI 兼容 API 地址 (默认 http://127.0.0.1:8080/v1)")
    p.add_argument("--model", default="local-ai", help="模型名 (默认 local-ai)")
    p.add_argument("-q", "--query", help="单次提问, 非交互模式")
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=0.7)
    return p


def api_call(url, model, messages, max_tokens, temperature, stream=True, timeout=600):
    body = json.dumps({
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": stream,
    }).encode("utf-8")
    req = urllib.request.Request(
        url.rstrip("/") + "/chat/completions", data=body,
        headers={"Content-Type": "application/json",
                 "User-Agent": "mini-agent/1.0"})
    return urllib.request.urlopen(req, timeout=timeout)


def stream_chat(url, model, messages, max_tokens, temperature):
    """流式对话, 打印并返回完整回复。"""
    resp = api_call(url, model, messages, max_tokens, temperature, stream=True)
    collected = []
    buf = b""
    for raw in resp:
        buf += raw
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            line = line.strip()
            if not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            if payload == b"[DONE]":
                break
            try:
                chunk = json.loads(payload)
                delta = chunk["choices"][0]["delta"].get("content")
                if delta:
                    print(delta, end="", flush=True)
                    collected.append(delta)
            except Exception:
                continue
    print()
    return "".join(collected)


def non_stream_chat(url, model, messages, max_tokens, temperature):
    resp = api_call(url, model, messages, max_tokens, temperature, stream=False)
    data = json.loads(resp.read())
    return data["choices"][0]["message"]["content"]


def server_ok(url):
    try:
        req = urllib.request.Request(url.rstrip("/") + "/models",
                                     headers={"User-Agent": "mini-agent/1.0"})
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status == 200
    except Exception:
        return False


def main():
    args = build_parser().parse_args()

    if not server_ok(args.url):
        print(f"[错误] 本地服务不可用: {args.url}")
        print("请先运行 start-server.cmd 启动大模型服务。")
        sys.exit(1)

    if args.query:
        messages = [{"role": "user", "content": args.query}]
        # 单次模式: 流式输出本身已打印, 不再重复打印返回值
        stream_chat(args.url, args.model, messages, args.max_tokens, args.temperature)
        return

    print("=" * 56)
    print("  Local AI 轻量助手 (本地大模型, 无需联网)")
    print(f"  模型: {args.model}   服务: {args.url}")
    print("  命令: /new 清空对话  /exit 退出")
    print("=" * 56)
    messages = []
    while True:
        try:
            user = input("\n你 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见!")
            break
        if not user:
            continue
        if user in ("/exit", "/quit", "/q"):
            print("再见!")
            break
        if user == "/new":
            messages = []
            print("[已清空对话历史]")
            continue
        messages.append({"role": "user", "content": user})
        print("AI > ", end="", flush=True)
        try:
            reply = stream_chat(args.url, args.model, messages,
                                args.max_tokens, args.temperature)
        except Exception as e:
            print(f"\n[错误] {e}")
            messages.pop()
            continue
        messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
