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
tiny launcher script resolves secrets from 1Password at runtime using `op read`
— nothing sensitive lives on disk. Survives Hermes updates, works in scripts,
cron, and the gateway. **No `op run` required** — avoids the environment
stripping that breaks terminal display.

```bash
# After setup, your .env goes from this:
DEEPSEEK_API_KEY=sk-abc123def456...

# To this (commented out):
# DEEPSEEK_API_KEY=sk-abc123def456...

# And op.env has only references:
DEEPSEEK_API_KEY=op://AgentShare/Deepseek Agent API key/credential
```

---

## Why This Matters

Hermes Agent stores API keys in `~/.hermes/.env` by default — they sit on disk
in plaintext. If someone reads that file, they have your keys permanently.
Rotating them means hunting down every copy.

1Password stores secrets encrypted at rest. By pointing Hermes at `op://`
references instead of raw keys, the actual secrets never touch a file. You can
rotate keys in 1Password without touching Hermes's configuration.

### Why Not `op run`?

The 1Password CLI's `op run` command seems like the natural tool — it resolves
`op://` references and injects them as environment variables. But in practice,
`op run` has two problems that break Hermes:

1. **Strips environment variables:** `op run` only passes through the variables
   listed in your env-file. It drops `$TERM`, `$LINES`, `$COLUMNS`, `$PATH`,
   and terminal control sequences. Hermes's TUI renders broken and ugly.

2. **Infinite recursion risk:** If the launcher path resolves back to itself,
   each invocation spawns another, creating an infinite chain.

**The fix:** Instead of `op run`, the launcher pre-resolves each secret with
`op read`, exports them into the current shell environment, then `exec`s the
real Hermes binary. All existing env vars are preserved. No recursion possible.

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

## Step 5: Create the Launcher Script

The launcher is a single bash script that pre-resolves secrets with `op read`,
exports them into the environment, then `exec`s the real Hermes binary. Place it
at `~/.hermes/bin/hermes` and symlink it into `~/.local/bin/hermes` (ahead of
the real binary in PATH).

```bash
# Create the launcher
mkdir -p ~/.hermes/bin
cat > ~/.hermes/bin/hermes << 'SCRIPT'
#!/usr/bin/env bash
# Hermes launcher — resolves 1Password secrets at runtime.
# No plaintext API keys on disk. No op run env mangling.
#
# Uses op read to pre-resolve each secret reference, exports them
# into the current shell, then execs the real Hermes binary.
# All existing env vars ($TERM, $PATH, $HOME, $LINES, $COLUMNS)
# are preserved — the TUI renders normally.

set -euo pipefail

REAL_HERMES="${HERMES_REAL_BIN:-$HOME/.hermes/hermes-agent/venv/bin/hermes}"
OP_ENV_FILE="$HOME/.hermes/op.env"

resolve_secrets() {
    local tmpfile
    tmpfile=$(mktemp) || return 1

    while IFS='=' read -r key ref; do
        [[ "$key" =~ ^[[:space:]]*# ]] && continue
        [[ -z "$key" ]] && continue
        key=$(echo "$key" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
        ref=$(echo "$ref" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')

        value=$(op read "$ref" 2>/dev/null) || {
            echo "WARNING: Failed to resolve $ref (key: $key)" >&2
            continue
        }
        printf '%s=%s\n' "$key" "$value"
    done < "$OP_ENV_FILE" > "$tmpfile"

    echo "$tmpfile"
}

if [ -f "$OP_ENV_FILE" ] && command -v op &>/dev/null; then
    SECRETS_FILE=$(resolve_secrets)
    trap "rm -f '$SECRETS_FILE'" EXIT
    export $(cat "$SECRETS_FILE" | xargs)
fi

exec "$REAL_HERMES" "$@"
SCRIPT

chmod +x ~/.hermes/bin/hermes

# Replace any existing symlink with the new launcher
rm -f ~/.local/bin/hermes
ln -s ~/.hermes/bin/hermes ~/.local/bin/hermes
```

**How it works:**

```
You type:  hermes auth list   (or any hermes command)
              ↓
~/.local/bin/hermes → ~/.hermes/bin/hermes  (launcher)
              ↓
op read "op://AgentShare/Deepseek Agent API key/credential"
  → sk-abc123...   (resolved from 1Password, never written to disk)
              ↓
export DEEPSEEK_API_KEY=sk-abc123...
export ELEVENLABS_API_KEY=sk-xyz789...
              ↓
exec /real/path/to/hermes "$@"    ← Hermes sees normal env vars,
                                     $TERM, $PATH, $HOME all intact
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
# Hermes 1Password secret mappings — resolved at launch
# Format: ENV_VAR=op://Vault/Item/field

DEEPSEEK_API_KEY=op://AgentShare/Deepseek Agent API key/credential
OPENROUTER_API_KEY=op://AgentShare/OpenRouter API key/credential
ANTHROPIC_API_KEY=op://AgentShare/Anthropic API key/credential
ELEVENLABS_API_KEY=op://AgentShare/Eleven Labs API Key for Agents/credential
# ... add more as you create 1Password items
EOF
```

> **This file contains only `op://` references — never real keys.** Even if
> someone reads it, they get a pointer, not a secret.

---

## Step 7: Migrate Existing Keys

After creating your 1Password items and `op.env`, comment out the old plaintext
keys in `~/.hermes/.env`:

```bash
# Comment out each key (don't delete — keeps a record of what was there)
sed -i '' 's|^DEEPSEEK_API_KEY=|# DEEPSEEK_API_KEY=|' ~/.hermes/.env
sed -i '' 's|^ELEVENLABS_API_KEY=|# ELEVENLABS_API_KEY=|' ~/.hermes/.env
# ... repeat for each key
```

The Hermes Agent can help with this. Ask it:

> "List the API keys currently in plaintext in my .env, and for each one, tell
> me the exact `op://` reference I should add to op.env after I create the
> corresponding 1Password item."

---

## Step 8: Test

```bash
# Should show your DeepSeek credential resolved from 1Password
hermes auth list
# deepseek (1 credentials):
#   #1  DEEPSEEK_API_KEY  api_key  env:DEEPSEEK_API_KEY ←

# Gateway continues working transparently
hermes gateway status
```

If you see `env:DEEPSEEK_API_KEY` with the `←` arrow, the injection is
working — the key came from 1Password, not from your `.env` file.

To double-check that no plaintext remains:

```bash
# Should show only the commented-out lines (or no matches)
grep -n 'DEEPSEEK_API_KEY' ~/.hermes/.env
```

---

## What Survives

| Scenario | Works? |
|----------|--------|
| `hermes update` | ✓ Launcher at `~/.hermes/bin/`, untouched |
| Shell switch (bash → zsh → fish) | ✓ No alias or dotfile dependency |
| Scripts / Makefiles / CI | ✓ `~/.local/bin` is in PATH |
| Gateway systemd service | ✓ Unit calls `hermes` → launcher intercepts |
| Sub-agent delegation | ✓ Non-interactive shells work |
| Changing API keys in 1Password | ✓ No file changes needed — just restart Hermes |
| TUI display | ✓ All terminal env vars preserved |

---

## Architecture

```
~/.local/bin/hermes              symlink to launcher (first in PATH)
        │
        ▼
~/.hermes/bin/hermes             bash launcher
        │
        ├─ reads ~/.hermes/op.env       VAR=op://ref mappings (pointers only)
        ├─ op read each reference       resolved from 1Password at runtime
        ├─ exports into shell env       DEEPSEEK_API_KEY=sk-abc...
        │                               $TERM, $PATH, $HOME preserved
        └─ exec                         │
                ▼                       │
        ~/.hermes/hermes-agent/venv/bin/hermes   (untouched real binary)
        ← secrets injected as normal env vars, no op run mangling
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
readlink ~/.local/bin/hermes  # must point to ~/.hermes/bin/hermes
echo $PATH | tr ':' '\n' | head -3   # ~/.local/bin must be early
```

**Weird TUI display / recursion with old `op run` approach:**
If you previously used the `op run`-based launcher and see infinite recursion
or broken terminal rendering, check that you're using the `op read`-based
launcher from this updated tutorial. The old approach strips terminal
environment variables.

**Secret not resolving:**
```bash
# Test resolution manually
export OP_SERVICE_ACCOUNT_TOKEN="$(grep OP_SERVICE_ACCOUNT_TOKEN ~/.hermes/.env | cut -d= -f2-)"
op read "$(grep 'DEEPSEEK_API_KEY' ~/.hermes/op.env | cut -d= -f2-)"
# Should output your actual API key
```

---

*Launcher source: [GitHub Gist](https://gist.github.com/linxichen/e7efe7344ed6638203b9e00e6ba52f4d)*
