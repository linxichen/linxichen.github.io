---
title: "给 Hermes Agent 接入火山引擎豆包语音：中文 TTS/STT 插件教程"
date: 2026-05-01T00:45:00-04:00
categories:
  - Tutorial
tags:
  - hermes-agent
  - volcengine
  - tts
  - stt
  - 火山引擎
  - 豆包
  - 语音合成
  - ai-agent
---

[Hermes Agent](https://github.com/NousResearch/hermes-agent) 是一个开源的 AI Agent 框架，支持在 Discord、Telegram、iMessage 等平台上运行。默认自带 Edge TTS 和 local whisper，但要获得**高质量中文语音**，火山引擎的豆包语音是目前的最优解。

本文记录完整接入过程——从获取 API Key 到 Discord 语音频道里开口说话。

## 为什么选火山引擎

| 对比 | Edge TTS (默认) | 火山引擎豆包 |
|------|----------------|-------------|
| 中文音色 | 2-3 种 | 60+ 种（情感、方言、角色） |
| 语音质量 | 一般 | 接近真人，2.0 模型支持情感变化 |
| API 稳定性 | 偶有波动 | 企业级 SLA |
| 中文识别 | 不支持 | 支持方言、说话人分离 |

## 步骤一：获取 API Key

⚠️ **关键区别**：火山引擎有**两个控制台**，Key 不通用。

1. 打开 [火山引擎语音控制台](https://console.volcengine.com/speech)（**不是** Ark/大模型控制台）
2. 左侧菜单 → API Key 管理 → 创建 API Key
3. 复制生成的 Key（格式类似 `a1a1d35f-38e4-4028-a2c5-02e145168b80`）

> Ark 平台的 Key（前缀 `ark-`）**不能**用于语音 API。如果用错了，会报 `45000010: Invalid X-Api-Key`。

## 步骤二：安装插件

插件已开源在 GitHub：

```bash
git clone https://github.com/linxichen/hermes-volcengine-voice.git \
  ~/.hermes/plugins/volcengine-voice
```

## 步骤三：配置

### 设置 API Key

```bash
echo 'VOLCENGINE_VOICE_API_KEY=你的API-Key' >> ~/.hermes/.env
```

### 切换到火山引擎

```bash
# TTS 文字转语音
hermes config set tts.provider volcengine
# STT 语音转文字
hermes config set stt.provider volcengine
```

### 选择音色（可选）

默认使用"爽快思思 2.0"，想换的话：

```bash
# 查看可用音色
# zh_female_conversation  — 爽快思思 2.0（默认）
# zh_female_gentle        — Vivi 2.0 温柔女声
# zh_female_sweet         — 甜美小源 2.0
# zh_male_ruya            — 儒雅逸辰 2.0

hermes config set tts.volcengine.speaker zh_female_gentle
```

### 重启

```bash
hermes gateway restart
```

## 步骤四：启用语音模式

在 Discord 里发送：

```
/voice tts
```

之后 Hermes 的所有回复都会用火山引擎合成语音说出来。

想加入语音频道实时对话：

1. 先进入一个 Discord 语音频道
2. 在文字频道发送 ``/voice channel``
3. Hermes 会自动加入并开始语音交互
4. 离开：``/voice leave``

## 插件架构

插件通过 monkey-patch 方式拦截 Hermes 的 TTS/STT 分发逻辑，不需要修改 Hermes 源码：

```
volcengine-voice/
├── plugin.yaml       # 插件元数据
├── __init__.py       # Monkey-patch 分发入口
├── tts.py            # HTTP Chunked TTS（V3 API）
└── stt.py            # WebSocket 二进制 STT（V3 API）
```

核心原理——在 ``register()`` 里包装 Hermes 的 ``text_to_speech_tool``：

```python
import tools.tts_tool as tts_module
original_tts = tts_module.text_to_speech_tool

def patched_tts(text, output_path=None):
    config = tts_module._load_tts_config()
    if tts_module._get_provider(config) == "volcengine":
        from hermes_plugins.volcengine_voice.tts import _volcengine_tts
        return _volcengine_tts(text, output_path, config)
    return original_tts(text, output_path)

tts_module.text_to_speech_tool = patched_tts
```

## 踩过的坑

1. **Ark Key vs Speech Key**：火山引擎的大模型平台（Ark）和语音平台是**独立产品**，Key 不互通。用 Ark Key 调语音 API 会 401。

2. **音色版本**：1.0 音色（如 `zh_female_shuangkuaisisi_moon_bigtts`）需要 `resource_id: seed-tts-1.0`，2.0 音色（如 `zh_female_shuangkuaisisi_uranus_bigtts`）需要 `resource_id: seed-tts-2.0`。混用会报 `55000000: resource ID mismatched`。

3. **Config 里的 speaker 是短名**：config.yaml 配置项会自动通过 VOICES 字典解析为完整 voice_type，不要直接写 `zh_female_shuangkuaisisi_uranus_bigtts`。

4. **Plugin import 路径**：Hermes 插件加载器使用 ``hermes_plugins`` 命名空间。插件内部导入 TTS/STT 模块时必须用 ``from hermes_plugins.volcengine_voice.tts import ...``，不能用 ``from plugins.volcengine_voice...``。

## 效果

配置完成后，所有平台——Discord、Telegram、iMessage——的语音交互都会自动走火山引擎。实测中文合成质量远超 Edge TTS，识别准确率也很高，支持中英混合。

## 参考

- [Hermes Agent 官方文档](https://hermes-agent.nousresearch.com/docs/)
- [火山引擎音色列表](https://www.volcengine.com/docs/6561/1257544)
- [插件 GitHub 仓库](https://github.com/linxichen/hermes-volcengine-voice)
