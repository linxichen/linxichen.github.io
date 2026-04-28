---
title: "Zero Plaintext Secrets on Disk: Wire Hermes Agent to 1Password with a Survivable Launcher"
date: 2026-04-28T09:00:00-04:00
categories:
  - Tutorial
tags:
  - Hermes Agent
  - 1Password
  - API keys
  - secrets management
  - devops
  - security
---

## TL;DR

Replace plaintext API keys in `~/.hermes/.env` with `op://` secret references. A
tiny launcher script wraps the `hermes` command so 1Password resolves keys at
runtime — nothing sensitive lives on disk. Survives Hermes updates, works in
scripts, cron, and the gateway.

```bash
# After setup, your .env goes from this:
DEEPSEEK_API_KEY=sk-abc123def456...

# To this (in op.env):
DEEPSEEK_API_KEY=op://AgentShare/Deepseek Agent API key/credential
```

---

## Why

Hermes Agent stores API keys in `~/.hermes/.env` by default — they sit on disk
in plaintext. If someone reads that file, they have your keys permanently.
Rotating them means hunting down every copy.

1Password stores secrets encrypted at rest. By pointing Hermes at `op://`
references instead of raw keys, the actual secrets never touch a file. You can
rotate keys in 1Password without touching Hermes's configuration.

The challenge: Hermes does not natively understand `op://` references. The
official 1Password CLI ships an `op run` command that resolves references and
injects them as environment variables before launching a command. This tutorial
wires that into a transparent launcher — you keep typing `hermes` and everything
just works.

---

## Step 1: Create a 1Password Service Account

Service accounts let machines authenticate to 1Password without an interactive
desktop app.

1. Sign in to [1Password](https://my.1password.com)
2. Go to **Settings → Service Accounts**
3. Click **Create Service Account**
4. Give it a name (e.g. "Hermes Agent")
5. Grant it **read access** to the vault where your API keys live
6. Copy the token — it starts with `ops_`

> **Important:** The token is shown exactly once. Save it immediately.

---

## Step 2: Store the Token in Hermes's Environment

Add the service account token to `~/.hermes/.env` so the `op` CLI can
authenticate:

```bash
echo 'OP_SERVICE_ACCOUNT_TOKEN=ops_your_token_here' >> ~/.hermes/.env
```

Verify connectivity:

```bash
export OP_SERVICE_ACCOUNT_TOKEN="$(grep OP_SERVICE_ACCOUNT_TOKEN ~/.hermes/.env | cut -d= -f2-)"
op whoami
# Should show: User Type: SERVICE_ACCOUNT
```

---

## Step 3: Install the 1Password CLI

```bash
# macOS
brew install 1password-cli

# Linux (Debian/Ubuntu)
curl -sS https://downloads.1password.com/linux/debian/amd64/stable/1password-cli-latest.deb -o /tmp/op.deb
sudo dpkg -i /tmp/op.deb

# Verify
op --version
```

---

## Step 4: Create API Credential Items in 1Password

For each API key you want to protect, create an **API Credential** item in your vault:

1. Open 1Password
2. Navigate to your vault (e.g. "AgentShare")
3. Click **New Item → API Credential**
4. Fill in:
   - **Title:** e.g. "Deepseek Agent API key"
   - **Credential:** paste the actual API key
5. Save

Note the secret reference path: `op://VaultName/Item Title/credential`

> **Service accounts are read-only by default.** If your service account cannot
> create items, either grant it write access in 1Password admin, use the desktop
> app to create items manually, or sign in interactively once with
> `op signin --account my.1password.com` to create them from the CLI.

---

## Step 5: Download the Launcher Script

The launcher is a single bash script that wraps `op run`. Drop it at
`~/.local/bin/hermes` — ahead of the real Hermes binary in PATH.

```bash
# Download from GitHub Gist
curl -fsSL https://gist.githubusercontent.com/linxichen/e7efe7344ed6638203b9e00e6ba52f4d/raw/hermes-launcher.sh -o ~/.local/bin/hermes
chmod +x ~/.local/bin/hermes
```

**How it works:**

```
You type:  hermes auth list   (or any hermes command)
              ↓
~/.local/bin/hermes  (launcher)
              ↓
op run --env-file=~/.hermes/op.env -- /real/path/to/hermes "$@"
              ↓                                    ↓
      1Password injects keys               Hermes picks them up
      into environment                     as normal env vars
```

The real Hermes binary (`~/.hermes/hermes-agent/venv/bin/hermes`) is never
touched. `hermes update` only touches files inside the venv — the launcher
survives.

---

## Step 6: Create Your op.env Mapping File

`~/.hermes/op.env` maps environment variable names to 1Password references.
One line per key:

```bash
cat > ~/.hermes/op.env << 'EOF'
# Hermes 1Password secret mappings — resolved by op run at launch
# Format: ENV_VAR=op://Vault/Item/field

DEEPSEEK_API_KEY=op://AgentShare/Deepseek Agent API key/credential
OPENROUTER_API_KEY=op://AgentShare/OpenRouter API key/credential
ANTHROPIC_API_KEY=op://AgentShare/Anthropic API key/credential
# ... add more as you create 1Password items
EOF
```

> **This file contains only `op://` references — never real keys.** Even if
> someone reads it, they get a pointer, not a secret.

---

## Step 7: Let Hermes Agent Help Wire Your Keys

The Hermes Agent already running on your machine (in the Discord or Telegram
gateway) can help you migrate existing keys. Just ask it:

> "List the API keys currently in plaintext in my .env, and for each one, tell
> me the exact `op://` reference I should add to op.env after I create the
> corresponding 1Password item."

The agent can read your `.env` file, identify active keys, and generate the
correct `op.env` lines for you. It can also verify that `op read` successfully
resolves each reference.

To remove a plaintext key from `.env` after wiring it:

```bash
# Comment out the key in .env (don't delete — keeps a record of what was there)
sed -i 's|^DEEPSEEK_API_KEY=|# DEEPSEEK_API_KEY=|' ~/.hermes/.env
```

---

## Step 8: Test

```bash
# Should show your DeepSeek credential resolved from 1Password
hermes auth list deepseek
# deepseek (1 credentials):
#   #1  DEEPSEEK_API_KEY  api_key  env:DEEPSEEK_API_KEY ←

# Gateway continues working transparently
hermes gateway status
```

If you see `env:DEEPSEEK_API_KEY` in the credential list, the injection is
working — the key came from 1Password, not from your `.env` file.

---

## What Survives

| Scenario | Works? |
|----------|--------|
| `hermes update` | ✓ Launcher in `~/.local/bin/`, untouched |
| Shell switch (bash → zsh → fish) | ✓ No alias or dotfile dependency |
| Scripts / Makefiles / CI | ✓ `~/.local/bin` is in PATH |
| Gateway systemd service | ✓ Unit calls `hermes` → launcher intercepts |
| Sub-agent delegation | ✓ Non-interactive shells work |
| Changing API keys in 1Password | ✓ No file changes needed — just restart Hermes |

---

## Architecture

```
~/.local/bin/hermes          bash launcher (first in PATH)
        │
        ▼
op run --env-file=~/.hermes/op.env -- <real hermes binary>
        │
        ├─ ~/.hermes/op.env         VAR=op://ref mappings (pointers, no keys)
        └─ ~/.hermes/.env           OP_SERVICE_ACCOUNT_TOKEN only
        │
        ▼
~/.hermes/hermes-agent/venv/bin/hermes    untouched real binary
```

---

## Troubleshooting

**`op` CLI not found:**
```bash
which op  # should return a path
```

**Service account can't read:**
```bash
op read "op://Vault/Item/field"
# If this fails, check the vault name, item name, and that the service account
# has access to that vault.
```

**Launcher not intercepting:**
```bash
which hermes        # must be ~/.local/bin/hermes
echo $PATH | tr ':' '\n' | head -3   # ~/.local/bin must be early
```

---

*Launcher source: [GitHub Gist](https://gist.github.com/linxichen/e7efe7344ed6638203b9e00e6ba52f4d)*
