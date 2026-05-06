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
echo 'VOLCENGINE_VOICE_API_KEY=你的API-Key' >> ~/.hermes/env.d/volcengine.env
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

## 真实调试记录：STT 不工作的 48 小时

教程写完了，插件发布了——然后用户在实际使用中发现：**语音频道里说话完全没反应**。

### 症状

- Bot 能正常加入语音频道 ✓
- Discord SPEAKING 事件正常触发 ✓
- 但转录结果始终为空，用户说的话石沉大海 ✗

### 排查过程

**第一步：确认音频采集正常**

检查 gateway 日志，SPEAKING 事件确实在触发：

```
SPEAKING event: ssrc=498 -> user=1490217802425172008
```

RTP 包正常解密，Opus 解码成功，PCM→WAV 转换也没问题。问题在 STT 环节。

**第二步：直接测试 Volcengine API**

绕开插件系统，写了一个裸 WebSocket 脚本直接连 `wss://openspeech.bytedance.com/api/v3/sauc/bigmodel_async`，逐字节 dump 服务端返回的二进制帧：

```
=== Response #2 ===
Header: 0x11931000
msg_type=0b1001  flags=0b0011  (LAST_NEGATIVE_SEQUENCE)
Payload size: 1  ← 只有 1 字节?!
Payload: 0x00     ← 空字节
```

1 字节的 payload？这不对。看原始 hex：

```
1193100000000001000000677b22617564696f5f696e666f223a7b...
```

仔细对照官方文档的二进制协议，发现了关键错误——

**第三步：找到根因——Sequence Number 偏移**

Volcengine 的 Full Server Response 格式是：

```
Header (4B) | Sequence (4B，当 flags=0b0001/0b0011) | PayloadSize (4B) | Payload
```

当服务端返回 `flags=0b0011`（最后一包，带负 sequence）时，Header 后的 4 字节是 **Sequence Number**（值 `0x00000001`），而不是 Payload Size。

插件代码把 Sequence 当成了 Size，读出来是 1，然后取了 1 字节 payload（一个空字节 `0x00`）。真正的 JSON payload（103 字节）被跳过了。

**第四步：第二个 Bug——响应 JSON 结构变化**

修复偏移后，成功拿到了 JSON：

```json
{"audio_info":{"duration":3000},"result":{"additions":{"log_id":"..."}}}
```

但还是没有文字！原来 v3 API 的转录文本在 `result.text`，而插件代码在找 `payload_msg.result`（旧版路径）。

### 修复

两个 commit 修复了 `stt.py`：

1. 根据 `flags` 判断是否有 Sequence 字段，动态调整 payload 偏移
2. 新增 `_extract_asr_text()` 函数，优先读取 v3 格式的 `result.text`，回退到旧格式

### 教训

1. **直接测 API，不要靠日志猜**。写一个裸 WebSocket 调试脚本，逐字节 hex dump，比看日志快 10 倍。
2. **二进制协议的"小字段"偏移错误**是最难发现的 bug——多读了 4 字节，整个响应就全乱套了。
3. **官方文档是最好的调试工具**。对照文档逐字段核对 header 的 bit layout，几分钟就定位到了问题。
4. **本地 STT 是保底方案**。调试期间切到 `stt.provider: local`（faster-whisper），不需要重启就能走通流程，验证音频采集和 silence detection 都没问题。

插件修复已推送到 [GitHub](https://github.com/linxichen/hermes-volcengine-voice)。

## 第二个调试回合：Voice Channel 里能听到我吗？

STT 协议修好了，以为万事大吉——结果用户一进语音频道，**连 SPEAKING 事件都没有**。

### 症状

- VoiceReceiver 正常启动 ✓
- SPEAKING hook 安装成功 ✓
- 但日志里没有任何 `SPEAKING event`，没有任何 UDP 包
- 5 分钟后自动超时断开（`VOICE_TIMEOUT=300`）

### 根因：Stale WebSocket

Gateway 进程一直没重启（从调试开始就运行着），Discord 语音频道的内部 WebSocket 连接在长时间无活动后变 stale。表面上连着，实际上收不到任何事件。

**解决**：`/voice leave` → `/voice channel` 重新加入，WebSocket 重建，问题消失。

### 延迟分析

修好后做了完整的 latency breakdown：

```
19:49:00.727  语音采集完成（含 1.5s 静音阈值判断）
19:49:01.140  STT 转录完成 ✅  (413ms！火山引擎确实快)
19:49:13.609  LLM 回复完成     (12.5s — deepseek-v4-pro)
19:49:15.040  TTS 开始播放
```

瓶颈在 LLM（12.5s），不是 STT。火山引擎 ASR 的 413ms 延迟完全够用。如果想进一步降低延迟，可以：

1. 把 voice channel 绑定的模型切到 `deepseek-chat`（更快但质量稍低）
2. 调低静音阈值 `silence_duration: 1.0`（从 3.0s 降到 1.0s）

### 修复：Voice 用户名显示

转录到的消息显示的是 raw Discord user ID（`1490217802425172008`）而不是用户名。Gateway 的 `_handle_voice_channel_input` 函数直接用了 `str(user_id)` 作为 `user_name`。修复通过 `guild.get_member()` 解析 display name：

```python
guild = adapter._client.get_guild(guild_id)
member = guild.get_member(user_id)
user_name = member.display_name if member else str(user_id)
```

这个修复也提交到了 hermes-agent 的 fork，等待 PR。

## 效果

配置完成后，所有平台——Discord、Telegram、iMessage——的语音交互都会自动走火山引擎。实测中文合成质量远超 Edge TTS，识别准确率也很高，支持中英混合。

## 参考

- [Hermes Agent 官方文档](https://hermes-agent.nousresearch.com/docs/)
- [火山引擎音色列表](https://www.volcengine.com/docs/6561/1257544)
- [插件 GitHub 仓库](https://github.com/linxichen/hermes-volcengine-voice)
