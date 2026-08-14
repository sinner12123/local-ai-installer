# -*- coding: utf-8 -*-
"""模型推荐引擎 — 根据硬件检测结果推荐最适配的本地大模型。

模型清单(2026-08 已通过 hf-mirror API 核实文件名与大小, 单位 GB):
  unsloth/Qwen3-1.7B-GGUF      Q4_K_M 1.11
  unsloth/Qwen3-4B-GGUF        Q4_K_M 2.50
  unsloth/Qwen3-8B-GGUF        Q4_K_M 5.03
  unsloth/Qwen3-14B-GGUF       Q4_K_M 9.00
  unsloth/Qwen3-32B-GGUF       Q4_K_M 19.76
  unsloth/Qwen3.5-35B-A3B-GGUF Q4_K_M 22.02 (MoE, 激活 3B, 快)

量化策略: 一般聊天 Q4_K_M 起步; 内存宽裕可上 Q5_K_M/Q6_K。
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import installer_core as core

MODELS = [
    {
        "id": "qwen3-1.7b", "name": "Qwen3-1.7B (轻量, 老旧电脑)",
        "repo": "unsloth/Qwen3-1.7B-GGUF",
        "file": "Qwen3-1.7B-Q4_K_M.gguf", "size_gb": 1.11,
        "min_ram_gb": 4, "min_vram_gb": 0, "desc": "老电脑/极低内存首选",
    },
    {
        "id": "qwen3-4b", "name": "Qwen3-4B (入门, CPU也能跑)",
        "repo": "unsloth/Qwen3-4B-GGUF",
        "file": "Qwen3-4B-Q4_K_M.gguf", "size_gb": 2.50,
        "min_ram_gb": 8, "min_vram_gb": 3, "desc": "无独显/小显存主力",
    },
    {
        "id": "qwen3-8b", "name": "Qwen3-8B (推荐, 平衡之选)",
        "repo": "unsloth/Qwen3-8B-GGUF",
        "file": "Qwen3-8B-Q4_K_M.gguf", "size_gb": 5.03,
        "min_ram_gb": 16, "min_vram_gb": 6, "desc": "8G显存或32G内存电脑首选",
    },
    {
        "id": "qwen3-14b", "name": "Qwen3-14B (高智商, 需大显存)",
        "repo": "unsloth/Qwen3-14B-GGUF",
        "file": "Qwen3-14B-Q4_K_M.gguf", "size_gb": 9.00,
        "min_ram_gb": 32, "min_vram_gb": 10, "desc": "12G+显存, 更强推理",
    },
    {
        "id": "qwen3-32b", "name": "Qwen3-32B (旗舰, 需24G显存)",
        "repo": "unsloth/Qwen3-32B-GGUF",
        "file": "Qwen3-32B-Q4_K_M.gguf", "size_gb": 19.76,
        "min_ram_gb": 64, "min_vram_gb": 22, "desc": "24G显存工作站",
    },
    {
        "id": "qwen3.5-35b-a3b", "name": "Qwen3.5-35B-A3B MoE (高速旗舰)",
        "repo": "unsloth/Qwen3.5-35B-A3B-GGUF",
        "file": "Qwen3.5-35B-A3B-Q4_K_M.gguf", "size_gb": 22.02,
        "min_ram_gb": 48, "min_vram_gb": 20, "desc": "MoE仅激活3B, 速度接近8B但智商接近32B",
    },
]

# 备用量化(内存紧张时降级用)
FALLBACK_QUANTS = {
    "qwen3-4b":  ("Qwen3-4B-Q3_K_M.gguf", 2.08),
    "qwen3-8b":  ("Qwen3-8B-Q3_K_M.gguf", 4.12),
    "qwen3-14b": ("Qwen3-14B-Q3_K_M.gguf", 7.3),  # 估算, 安装时以 API 为准
}


def recommend(hw):
    """根据硬件返回推荐模型列表 (从最适配到次适配)。

    返回: [{"model": <MODELS条目>, "engine": "cuda"|"vulkan"|"cpu",
            "ngl": int, "ctx": int, "reason": str, "fit": "exact"|"tight"|"loose"}]
    """
    gpus = hw.get("gpus", [])
    ram_gb = hw.get("ram", {}).get("total_gb", 0)
    free_disk = max((d["free_gb"] for d in hw.get("disks", [])), default=0)

    nvidia = [g for g in gpus if g["vendor"] == "nvidia"]
    amd = [g for g in gpus if g["vendor"] == "amd"]
    intel = [g for g in gpus if g["vendor"] == "intel"]

    vram = max((g["vram_gb"] for g in gpus), default=0)
    engine, ngl = "cpu", 0

    if nvidia and vram >= 3:
        # Blackwell (RTX 50系) 需要 CUDA 13.3 构建 + 新驱动; 驱动不够新回退 Vulkan
        nv = nvidia[0]
        if core.is_blackwell(nv.get("name", "")) and not core.driver_supports_cuda_13_3(nv.get("driver", "")):
            engine, ngl = "vulkan", 999
        else:
            engine, ngl = "cuda", 999          # 全层进 GPU
    elif amd or intel:
        engine, ngl = "vulkan", 999
    # CPU 纯推: engine="cpu", ngl=0

    results = []
    used = set()

    def try_model(mid, need_vram, need_ram, reason, fit):
        if mid in used:
            return
        m = next((x for x in MODELS if x["id"] == mid), None)
        if not m:
            return
        if engine != "cpu" and vram < need_vram:
            return
        if ram_gb < need_ram:
            return
        if free_disk and free_disk < m["size_gb"] + 4:
            return
        used.add(mid)
        results.append({"model": m, "engine": engine, "ngl": ngl,
                        "ctx": 8192, "reason": reason, "fit": fit})

    if engine != "cpu":
        # 有 GPU
        if vram >= 22 and ram_gb >= 48:
            try_model("qwen3.5-35b-a3b", 20, 48, "MoE 旗舰, 24G 显存可全量载入", "exact")
        if vram >= 22 and ram_gb >= 64:
            try_model("qwen3-32b", 22, 64, "32B 旗舰, 24G 显存可全量载入", "exact")
        if vram >= 10 and ram_gb >= 32:
            try_model("qwen3-14b", 10, 32, "14B 全量进 GPU", "exact")
        if vram >= 6 and ram_gb >= 16:
            try_model("qwen3-8b", 6, 16, "8B 全量进 GPU, 速度与质量均衡", "exact")
        if vram >= 3.5 and ram_gb >= 8:
            try_model("qwen3-4b", 3.5, 8, "4B 全量进 GPU", "exact")
        # 显存略紧: 次档
        if vram >= 4 and ram_gb >= 16 and "qwen3-8b" not in used:
            try_model("qwen3-8b", 4, 16, "显存偏紧, 8B 部分层进 GPU(自动)", "tight")
        if not results:
            try_model("qwen3-4b", 3, 8, "显存不足, 4B 保底", "tight")
    else:
        # 纯 CPU
        if ram_gb >= 32:
            try_model("qwen3-8b", 0, 32, "无独显但内存大, 8B CPU 推理(慢但可用)", "tight")
        if ram_gb >= 16:
            try_model("qwen3-4b", 0, 16, "无独显, 4B CPU 推理", "exact")
        if ram_gb >= 8:
            try_model("qwen3-4b", 0, 8, "无独显, 4B CPU 推理", "tight")
        if ram_gb >= 4:
            try_model("qwen3-1.7b", 0, 4, "内存紧张, 1.7B 保底", "exact")

    # 永远兜底 1.7B
    if not results:
        try_model("qwen3-1.7b", 0, 4, "最后保底方案", "loose")

    # 按 fit 排序: exact > tight > loose
    order = {"exact": 0, "tight": 1, "loose": 2}
    results.sort(key=lambda r: order.get(r["fit"], 9))
    return results


def pick_default(recos):
    """返回第一个推荐 (最适配)。"""
    return recos[0] if recos else None


if __name__ == "__main__":
    import hardware
    hw = hardware.detect_all()
    print("硬件:", json_dump(hw))
    for r in recommend(hw):
        m = r["model"]
        print(f"  -> {m['name']} [{r['engine']}] {m['size_gb']}GB  ({r['reason']})")


def json_dump(o):
    import json
    return json.dumps(o, ensure_ascii=False)
