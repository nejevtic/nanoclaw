#!/usr/bin/env bash
# Setup helper: install-node — bundles Node 22 install into one idempotent
# script so /new-setup can run it without needing `curl | sudo -E bash -` in
# the allowlist (that pattern is inherently unmatchable — bash reads from
# stdin, so pre-approval can't inspect what's being executed).
#
# The script itself is the allowlisted unit; the pipes and sudo live inside
# it. Pure bash by design — runs before Node exists on the host.
set -euo pipefail

echo "=== NANOCLAW SETUP: INSTALL_NODE ==="

if command -v node >/dev/null 2>&1; then
  echo "STATUS: already-installed"
  echo "NODE_VERSION: $(node --version)"
  echo "=== END ==="
  exit 0
fi

if command -v uvx >/dev/null 2>&1; then
  echo "STEP: uvx-nodeenv"
  uvx nodeenv -n lts ~/node
  mkdir -p ~/.local/bin
  ln -sf ~/node/bin/node ~/.local/bin/node
  ln -sf ~/node/bin/npm ~/.local/bin/npm
  ln -sf ~/node/bin/npx ~/.local/bin/npx
  ln -sf ~/node/bin/pnpm ~/.local/bin/pnpm
else
  case "$(uname -s)" in
    Darwin)
      echo "STEP: brew-install-node"
      if ! command -v brew >/dev/null 2>&1; then
        echo "STATUS: failed"
        echo "ERROR: Homebrew not installed. Install brew first (https://brew.sh) then re-run."
        echo "=== END ==="
        exit 1
      fi
      brew install node@22
      ;;
    Linux)
      # apt (and the nodesource script it drives) runs debconf, which on a
      # live tty renders interactive dialogs — e.g. a "Pending kernel
      # upgrade" <Ok> prompt — that block the whole setup until a human
      # answers them. Non-interactive debconf always falls back to defaults,
      # so scripted installs never wait on a terminal. `-E` carries the var
      # into the nodesource script; `env` forces it past sudo's env_reset
      # for the direct apt-get call.
      export DEBIAN_FRONTEND=noninteractive
      echo "STEP: nodesource-setup"
      curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash -
      echo "STEP: apt-install-nodejs"
      sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y nodejs
      ;;
    *)
      echo "STATUS: failed"
      echo "ERROR: Unsupported platform: $(uname -s)"
      echo "=== END ==="
      exit 1
      ;;
  esac
fi

if ! command -v node >/dev/null 2>&1; then
  echo "STATUS: failed"
  echo "ERROR: node not found on PATH after install"
  echo "=== END ==="
  exit 1
fi

echo "STATUS: installed"
echo "NODE_VERSION: $(node --version)"
echo "=== END ==="
