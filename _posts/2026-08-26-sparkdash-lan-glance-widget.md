---
title: "sparkDash on the LAN + a Glance widget for it"
date: 2026-08-26T14:00:00-04:00
categories:
  - Notes
tags:
  - homelab
  - dgx-spark
  - glance
  - monitoring
  - sparkdash
---

Deployed [sparkDash](https://github.com/MiaAI-Lab/sparkDash) (multi-DGX-Spark monitoring) to the head Spark node and wired it into Glance. Notes to self so I remember the shape.

## What I did

- **Host**: `spark1` (192.168.1.205, aarch64). The container is arm64-only, so it can't live on homelab-network (x86_64) — it runs on the head Spark itself.
- **Deploy**: `git clone` + `docker compose up --build`. Compose uses `network_mode: host`, `privileged: true`, `pid: host`, and mounts `/proc`/`/sys`/`/` + `nvidia-smi` so it reads host metrics directly. Container name `sparkDash`, `restart: always`, listens on `:5555`.
- **Units**: added via the REST API — `spark1 (head)` is local collectors + LLM probe on `:8888` (deepseek-v4-flash-dspark), `spark2 (worker)` is remote over SSH with the password stored encrypted (`config/sparks-secrets.json`). Node config lives in `config/sparks.json` inside a Docker volume, not git.
- **DNS/reverse-proxy**: appended a `sparkdash.linxic.com:443` block to caddy-internal on npm-intra → `192.168.1.205:5555`, then `caddy reload`. Cert auto-issued via Cloudflare DNS challenge. Live at `https://sparkdash.linxic.com`.

## Glance

- **Bookmark** in the homelab page under an "AI / Compute" group.
- **Live widget**: a `custom-api` widget (`glance/config/sparkdash.yml`) that uses `subrequests` to fetch `/api/sparks/spark1/metrics` and `/api/sparks/spark2/metrics` concurrently, rendering GPU temp/util, VRAM, tok/s, KV-cache %, model, uptime per node. `cache: 30s`.

Gotchas that cost me time:

- Go templates don't allow `.Subrequest "x".JSON` — assign to a var first: `{{ $s2 := .Subrequest "spark2" }}` then `$s2.JSON.Int "..."`.
- `div` wants exactly 2 args — wrap the whole method chain in parens (`div (.JSON.Int "x") 1024`), or it reads 3 args.
- The `/api/sparks` list endpoint returns *config only* (no live metrics); the per-id `/api/sparks/:id/metrics` is the one with GPU/LLM numbers.

## Published

- Glance community widget PR: `glanceapp/community-widgets#311` (portable version using `${SPARKDASH_*_URL}` env vars, no hardcoded IPs).
- Notified the sparkDash repo: `MiaAI-Lab/sparkDash#66`.
