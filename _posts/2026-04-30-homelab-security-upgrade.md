---
title: "Homelab 全栈安防升级方案：Alarmo + Frigate + AI Agent 从零到一"
date: 2026-04-30T15:00:00-04:00
categories:
  - Tutorial
tags:
  - Home Assistant
  - homelab
  - security
  - AI agent
  - Frigate
  - Alarmo
  - camera
  - smart home
---

## TL;DR

如果你有 homelab（NAS + GPU 服务器 + Home Assistant），你已经有了搭建一套全本地、AI 驱动的家庭安防系统所需的基础设施。缺的只是传感器、一个软件层（Alarmo），以及可选的电话报警服务（Noonlight）。这篇从零到一讲清楚每一个组件。

---

## 背景：为什么 Wyze 不够用

我用的是 Wyze 摄像头 + Garage Cam。它在 Home Assistant 里能控制车库门，但有几个致命问题：

- ❌ **没有 camera 实体** — 无法在 HA 里看画面、snapshot、AI 分析
- ❌ **只有 cover/switch/sensor/siren** — 功能残缺
- ❌ **云端依赖** — Wyze 有多次数据泄露前科，所有视频流都要先走云端
- ⚠️ **车库门控制不可靠** — 发了打开指令返回 success，状态却还是 closed

结论：**Wyze 可以做辅助，但不能做安防核心。**

---

## 一、摄像头：换 Reolink，一步到位

Home Assistant 社区对主流摄像头有明确评级：

- 🥇 **Reolink** — HA 铂金评级，原生 RTSP/ONVIF，完全本地，$60 起，**社区公认最佳**
- 🥇 **UniFi Protect** — HA 金级，体验一流但锁定 UniFi 生态（需要 UDM/UNVR）
- 🥇 **Amcrest** — HA 金级，预算首选，原生 ONVIF
- 🥈 **Tapo (TP-Link)** — HA 银级，$30 入门，RTSP 支持
- ⚠️ **Wyze** — 不推荐继续投入（云依赖、隐私风险）

**推荐方案：**

- **户外** — Reolink RLC-810A（4K、PoE、人车检测，$60）
- **室内** — Reolink E1 Zoom（2K、PTZ，$40）
- **门铃** — Reolink Video Doorbell（PoE，$90）

如果你已有 Wyze 摄像头不想扔，可以刷 [docker-wyze-bridge](https://github.com/mrlt8/docker-wyze-bridge) 把 RTSP 流暴露给 Frigate 消费。但长期来看，换 Reolink 值得。

---

## 二、NVR + AI 检测：Frigate 是核心

### Frigate NVR — 本地 AI 对象检测的标准答案

[Frigate](https://github.com/blakeblackshear/frigate) 是一个开源 NVR，31.7k GitHub Stars，专为 Home Assistant 设计。它做的事情：

```
摄像头 RTSP 流
    → Frigate (TensorFlow / ONNX)
        → 人、车、动物、包裹、车牌
            → MQTT
                → Home Assistant 自动化触发
```

**关键特性：**

- **本地 AI 推理** — 支持 Google Coral TPU（USB $60），推理延迟 < 50ms
- **智能录制** — 有人才录，没人跳过；支持 24/7 全录
- **v0.17 新功能** — AI Review Summaries（Ollama 自动生成事件描述）、分类模型 UI 向导
- **零订阅** — MIT 开源

**跑在哪？**

你的 homelab 天然适配：

- NAS 上跑 Frigate Docker 容器  
- Coral USB TPU 加速推理（插 NAS 的 USB 口）  
- 录制的视频存 NAS 的 HDD（几十 TB 空间）  

```yaml
# docker-compose.yml 片段
frigate:
  image: ghcr.io/blakeblackshear/frigate:stable
  devices:
    - /dev/bus/usb:/dev/bus/usb  # Coral TPU
  volumes:
    - /mnt/tank/frigate:/media/frigate  # 视频存 NAS
    - ./config.yml:/config/config.yml
  ports:
    - "5000:5000"
```

---

## 三、报警系统：Alarmo 软件层

### Alarmo — 把你已有的传感器变成安防系统

[Alarmo](https://github.com/nielsfaber/alarmo) 是 HA 的事实标准报警集成（2.1k ⭐），安装只需两步：

1. HACS → 搜索 Alarmo → 安装
2. 浏览器 UI 配置，零 YAML

**五种布防模式：**

- `armed_away` — 全屋无人，所有传感器激活
- `armed_home` — 有人在家，只守周界（门窗）
- `armed_night` — 睡眠模式
- `armed_vacation` — 长假模式
- `armed_custom_bypass` — 自定义排除

**你需要新增的硬件（传感器 + 执行器）：**

- **门窗传感器** — Aqara Zigbee 门磁（$10/个），纽扣电池 2 年续航
- **人体传感器** — Aqara P1（$15）或 Apollo 毫米波存在传感器（$35，能检测静止人体）
- **烟雾/CO** — First Alert Z-Wave（$40）
- **警报器** — Aeotec Z-Wave Siren 6（$55），多种铃声，HA 直控

**协议选择：**

- **Z-Wave** — 更安全（强制认证），频率不干扰 Wi-Fi，报警系统首选
- **Zigbee** — 更便宜、设备更多、生态更大
- **两个都跑** — HA 同时支持

### 如果你家有 wired 传感器

你家有预埋网线 + PoE，有两个好选择：

- **[Hornet Nest Alarm Panel](https://hackaday.com/2025/01/04/poe-power-protection-the-hornet-nest-alarm-panel/)** — PoE 供电，42 个有线传感器输入区，原生 ESPHome → 直连 HA，€259
- **Kincony A16 ESP32 板** — 16 路有线输入，刷 ESPHome，$40

如果你有旧的报警面板（DSC/Honeywell/Qolsys），**[Konnected Panel Pro](https://konnected.io/)**（$129，12 区）可以把它变成智能面板，保留所有有线传感器。

---

## 四、AI Agent 层：让你的 GPU 服务器上场

### LLM Vision — 让 AI 看懂摄像头画面

[ha-llmvision](https://github.com/valentinfrlch/ha-llmvision) v1.6.0 是专为 HA 设计的视觉 AI 集成：

- `image_analyzer` — 分析单张/多张摄像头图片
- `stream_analyzer` — 分析实时摄像头流
- `video_analyzer` — 分析 Frigate 事件视频
- `data_analyzer` — 从图片提取结构化数据 → 更新 HA 实体
- **事件记忆时间线** — 自然语言查询"快递送到了吗？"

支持 Ollama / OpenAI / Anthropic / Google / OpenRouter。你的 GPU 服务器可以跑本地模型（gemma3:4b 或 llama3.2-vision），零数据外泄。

### 联动流程

```
门窗被打开
    → Alarmo 触发警报
    → Frigate 抓摄像头 snapshot
    → LLM Vision 分析（GPU 本地推理）：
      "门口有可疑人员，黑色卫衣，在门口徘徊超过 2 分钟"
    → Hermes Agent 推送 Discord：
      "🚨 警报！前门被打开。
       门口有疑似可疑人员，穿黑色卫衣。
       需要帮你报警吗？"
```

### AI 自动布防/撤防

```
手机 GPS 离开 home zone
    + Frigate 确认车库没人在活动
    → AI 决策：自动 armed_away
    → Discord 通知："检测到所有人已离开，已自动布防。"
```

### 双向 SMS 控制（可选）

借鉴 [LangGraph + FastMCP](https://pinnsg.com/giving-my-diy-home-alarm-a-voice-how-i-connected-a-raspberry-pi-security-system-to-an-llm-powered-sms-agent/) 架构：

```
你发 SMS："今晚帮我设防"
    → Twilio → LangChain Agent
    → Alarmo arm_night
    → 回复 SMS："已布防夜间模式。后门已锁，车库门已关。"
```

---

## 五、电话报警：Noonlight 专业监控

DIY 方案最大的缺失是**报警后没人帮你打电话**。

[Noonlight](https://www.noonlight.com/) 解决了这个问题：

- **工作流程：** Alarmo 触发 → Noonlight 收到信号 → 真人 dispatcher 发短信 + 打电话给你 → 你接电话报 PIN 码取消 / 不接 → dispatcher 帮你打 911
- 💰 $10/月，无合约
- ❌ 误报无惩罚（报 PIN 就行）
- 🏠 可出保险证明（home insurance discount）
- 🔌 通过 [noonlight-hass](https://github.com/konnected-io/noonlight-hass) 直连 HA → Alarmo

---

## 六、停电应对

- **UPS** — NAS + PoE 交换机 + 路由器 + ONT 全插 UPS
- **NUT（Network UPS Tools）** — HA 监控 UPS 状态，停电自动通知 + 安全关机
- **Zigbee/Z-Wave 电池传感器** — 不依赖市电
- **PoE 摄像头 + 交换机在 UPS 上** — 继续录像
- **4G 备份路由器**（$50-100，插 UPS）→ ISP 断了还有 cellular

---

## 七、购物清单

**最小可行方案（$250-350）：**

- Alarmo — 免费（HACS 安装）
- Reolink RLC-810A × 2 — $120
- Aqara 门磁 × 6 — $60
- Aqara 人体传感器 × 3 — $45
- Aeotec Siren 6 — $55
- Coral USB TPU — $60（用于 Frigate）
- Zigbee 适配器（如果没有）— $30

**进阶方案（$500-800）：**

- 基础方案全部
- Reolink 门铃 PoE — $90
- Apollo 毫米波传感器 × 3 — $105
- First Alert Z-Wave 烟雾/CO × 3 — $120
- 旧 iPad 挂墙当控制面板 — $50（二手）

**完整方案（$800+）：**

- 进阶方案全部
- Noonlight 订阅 — $10/月
- Hornet Nest Alarm Panel（有线传感器）— €259
- Konnected Panel Pro（替代旧面板）— $129

---

## 八、软件栈一览

```
┌─ 传感器层 ────────────────────────────────────┐
│  有线门磁/人体 → Hornet Nest (PoE) → ESPHome    │
│  Zigbee 门磁/人体 (电池)                         │
│  烟雾/CO → Z-Wave                              │
├─ NVR + AI 层 ──────────────────────────────────┤
│  Frigate (Docker on NAS, Coral TPU 加速)        │
│  LLM Vision (GPU 推理: gemma3 / llama3.2-vision)│
├─ 控制层 ────────────────────────────────────────┤
│  Home Assistant + Alarmo                         │
│  Hermes Agent (Discord/iMessage 通知 + 语音控制) │
├─ 执行层 ────────────────────────────────────────┤
│  Aeotec Siren 6 (Z-Wave)                       │
│  Noonlight ($10/月, 真人 dispatcher)            │
├─ 存储层 ────────────────────────────────────────┤
│  NAS HDD (30 天滚动录像)                         │
├─ 抗停电层 ──────────────────────────────────────┤
│  UPS + NUT → HA 停电检测 → 通知 + 安全关机       │
│  4G 备份路由器 (可选)                             │
└────────────────────────────────────────────────┘
```

---

## 总结

homelab 用户做安防最大的优势是：你已经有 NAS、GPU、HA、有线网、UPS 这些最难搞的基础设施。缺的只是传感器（几十块一个）和软件层（全免费开源）。组合起来就是一套全本地、AI 驱动、隐私优先、经得起停电考验的家庭安防系统。

**下一步：** 装 Alarmo（5 分钟）→ 买传感器（等快递）→ 装 Frigate → 接入 LLM Vision → 订阅 Noonlight。

有问题去 [Hermes Agent Discord](https://discord.gg/cAzWEnpS) 找我。

---

*— Linxi Chen, April 2026*
