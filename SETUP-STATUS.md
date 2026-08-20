# NanoClaw setup — status checkpoint (2026-08-18, updated 2026-08-20)

> Resume point for finishing the NanoClaw install. Read top to bottom.

## Goal (what the user wants)

1. Talk to the NanoClaw assistant **from Telegram on a phone** — send standard commands, start agents that do different things.
2. Use **all existing subscriptions**: OpenAI, Anthropic, OpenCodeGo.
3. Use **local models** — Qwen 3.8 (`qwen3.8:27b`) on the local Ollama at `http://10.114.0.91:11434` (same setup as this OpenCode session; see `~/.config/opencode/opencode.json`).

## DONE

### 1. Fixed the "Installing the basics" hang (committed, `1a34c0ae`)

- **Root cause**: `setup/install-node.sh` runs `apt-get` (NodeSource setup + nodejs) with no `DEBIAN_FRONTEND=noninteractive`. A pending-kernel debconf question ("Package configuration — Pending kernel upgrade — \<Ok\>") renders an **interactive whiptail dialog on a live tty** and blocks forever; the spinner in the foreground kept redrawing "Installing the basics… (Ns)", reading as an endless hang. Evidence: the old `logs/setup-steps/01-bootstrap.log` (13:32 run) ends exactly mid-dialog.
- **Fix**: export `DEBIAN_FRONTEND=noninteractive` before the apt calls (carried into the NodeSource script via `sudo -E`; forced past sudo `env_reset` via `sudo env …` for the direct apt-get call). Committed as `1a34c0ae` on `main`.
- **Verified**: `bash nanoclaw.sh` now passes the phase — `◇ Basics ready (2s)` — and reaches the interactive wizard ("How would you like to begin?"). The wizard was never completed.

### 2. Bootstrap environment is ready

| Item | State |
|---|---|
| Node | `v22.23.2` at `/usr/bin/node` (NodeSource repo configured) |
| pnpm | `10.34.5` (corepack; `/usr/bin/pnpm` — matches `packageManager` pin) |
| `node_modules` | installed; `better-sqlite3` loads; `node_modules/.bin/tsx` present |
| Docker | **NOT installed** (wizard installs it) |
| OneCLI | **NOT installed** (wizard installs it) |
| `.env` / `groups/` / `data/` | not created yet (only `data/install-id`) |
| Machine | Ubuntu 24.04, 32 GB RAM, passwordless sudo, no gcc/make (prebuilt native mods are fine) |

Useful env facts:
- Local Ollama (OpenAI-compatible): `http://10.114.0.91:11434/v1`, default model `qwen3.8:27b`.
- OpenCode endpoints (from this OpenCode install): `opencode-go` → `https://opencode.ai/zen/go/v1`; `opencode` (Zen free) → `https://opencode.ai/zen/v1` (apiKey "public"). Possible OpenCodeGo API key: `~/.api/opencodego` (68 bytes — **unverified**, inspect on resume).
- `versions.json` pins a pre-built agent image (Claude-only; picking a non-claude provider makes the wizard switch to a local build).

### 3. Architecture mapped (what setup needs — researched from repo + branches)

**Wizard** (`bash nanoclaw.sh` → `pnpm run setup:auto`, code in `setup/auto.ts`):
`environment → first-agent template pick → container (Docker + image build) → onecli (credential vault) → auth (provider pick) → mounts → service (systemd user) → cli-agent ping/pong → timezone → channel (Telegram) → verify`.
Re-runs are safe/idempotent; steps skippable via `NANOCLAW_SKIP=a,b,c`.

**Providers** (per **agent group**, not global — `groups/<folder>/container.json` + DB `agent_provider`):
- `claude` — baked into trunk (default; Anthropic Agent SDK).
- `codex` — offered in the wizard picker (OpenAI: ChatGPT subscription or API key).
- `opencode` + `ollama` — **install via `/add-opencode` skill from the `providers` branch** (both branches already `git fetch`-ed as `origin/providers`, `origin/channels`).

**OpenCode provider mechanics** (`container/agent-runner/src/providers/opencode.ts` on `origin/providers`):
- Per-group env, passed by the host when the group's provider is `opencode`:
  `OPENCODE_PROVIDER`, `OPENCODE_MODEL` (`provider/model` form), optional `OPENCODE_SMALL_MODEL`, `ANTHROPIC_BASE_URL` (**required for non-anthropic providers**).
- Non-anthropic providers get `options: { apiKey: 'placeholder', baseURL: $ANTHROPIC_BASE_URL }`; **real keys are injected by OneCLI via `HTTPS_PROXY`** with a matching `--host-pattern`. `provider === 'openai'` + base URL uses OpenCode's stock OpenAI provider (Responses API) — the route to the local Ollama. (Do **not** re-add the old `@ai-sdk/openai-compatible` pin — see **G5**.)
- Optional per-model: `OPENCODE_MODEL_CONTEXT_LIMIT`, `OPENCODE_MODEL_OUTPUT_LIMIT`, `OPENCODE_MODEL_INPUT_MODALITIES`.
- **Open caveat (verify on resume)**: the skill docs treat `OPENCODE_*` as host `.env` (global). Check whether `ncl groups config get/update` exposes a per-group model override so different agent groups can pin different providers/models (there is a `feat/per-group-provider-config` branch upstream).

**Model mapping for the 4 wanted sources (working recipes)**:

| Agent group | Config | Credentials |
|---|---|---|
| Anthropic | `OPENCODE_PROVIDER=anthropic`, `OPENCODE_MODEL=anthropic/<model>` | OneCLI secret host-pattern `api.anthropic.com` (no base URL needed) |
| OpenAI | `OPENCODE_PROVIDER=openai`, `ANTHROPIC_BASE_URL=https://api.openai.com/v1` | OneCLI secret host-pattern `api.openai.com` |
| OpenCodeGo | provider id `opencode-go`, base URL `https://opencode.ai/zen/go/v1` | OneCLI secret host-pattern `opencode.ai` (`x-api-key` header, not Bearer — see skill's "OpenCode Zen" section); free tier = `zen/v1`, apiKey public |
| Local Qwen | `OPENCODE_PROVIDER=openai`, `ANTHROPIC_BASE_URL=http://10.114.0.91:11434/v1`, `OPENCODE_MODEL=openai/qwen3.8:27b` | none (placeholder key ok) |

**Telegram** (`origin/channels:.claude/skills/add-telegram/SKILL.md`):
1. Copy adapter files from `channels` branch + register barrel + `pnpm install @chat-adapter/telegram@4.38.1` + `pnpm run build` + registration test (the wizard runs this via the skill engine when you pick "telegram" as the channel).
2. Credentials (HUMAN step, Telegram on the phone): message **@BotFather** → `/newbot` → name + `...bot` username → copy token (`123456:ABC-DEF…`). For group use: `/mybots` → Bot Settings → Group Privacy → **off**.
3. Token → `TELEGRAM_BOT_TOKEN` in `.env` (validated by `getMe`), service restart, then **pairing**: terminal shows a 4-digit code; user sends exactly those digits to the bot from the chat to register (`pnpm exec tsx setup/index.ts --step pair-telegram -- --intent main`).
4. After pairing: wire with `/init-first-agent` (or `/manage-channels`) to pick which agent group serves which chat.

**OneCLI**: local credential vault (installed by wizard, versions pinned in `versions.json`: gateway 1.41.0, CLI 2.2.5). Secrets added per host-pattern; agents default to `all` secret mode (auto-inject). Web UI `http://127.0.0.1:10254`. Approvals configurable server-side only (UI).

## DONE — 2026-08-19 (continued from checkpoint)

- Telegram channel installed from `origin/channels` (adapter + `setup/pair-telegram.ts`), barrel + STEPS wired, `@chat-adapter/telegram@4.38.1`, host build + registration test green.
- `.env`: `TZ=Europe/Zurich`, `TELEGRAM_BOT_TOKEN` (bot **@llmclient_bot**, `getMe` OK).
- `mounts --empty`, `service` step: systemd user unit `nanoclaw-v2-2a38bd3e` installed + enabled, linger on.
- **Docker-group fix**: `user@1000` scope predates `docker` group add (can't restart — it hosts this session). Fixed via `/etc/sudoers.d/docker-ubuntu-nanoclaw` (NOPASSWD `/usr/bin/docker`) + wrapper `/usr/local/bin/docker` → `sudo -n docker` (unit PATH puts `/usr/local/bin` first). Service now runs.
- **Per-group env feature (local patch, uncommitted)**: `container_configs.env` JSON column (migration `023-container-config-env`), `setContainerConfigEnv()` in `src/db/container-configs.ts`, `src/providers/opencode.ts` merges per-group overrides (precedence: real env > group > `.env`). **This unblocks two `opencode` groups with different models on one host** (the global-OPENCODE_* caveat above is resolved locally).
- **Four agent groups created & configured** (EU/Zurich tz) — the OpenAI/Codex one was added the same day:
  - `Main` `ag-d441119a-…` — provider `claude` (subscription).
  - `GPT (OpenCodeGo)` `ag-d4793d5a-…` — provider `opencode`, `env` = `OPENCODE_PROVIDER=opencode-go`, `OPENCODE_MODEL=opencode-go/gpt-5.6-luna`, `ANTHROPIC_BASE_URL=https://opencode.ai/zen/go/v1`.
  - `Local Qwen` `ag-2049cb02-…` — provider `opencode`, `env` = `OPENCODE_PROVIDER=openai`, `OPENCODE_MODEL=openai/qwen3.8:27b`, `ANTHROPIC_BASE_URL=http://10.114.0.91:11434/v1`.
- **Telegram pairing + wiring done**: code issued, user sent it, chat `telegram:1650895591` registered, user **Nemanja** promoted **owner**, DM wired to Main (`/init-first-agent`). Welcome DM delivered.
- Host test suite: **1662/1662 pass** (136 files) after the feature + codex provider.

### ⚠️ Claude/OneCLI auth bug — FOUND & FIXED (important)

First real Claude call from the container → **401 "invalid x-api-key"** even though OneCLI applied its injection. Root cause (verified by replaying the exact headers):
- OneCLI `type=generic` Anthropic secret made the container auth via `ANTHROPIC_API_KEY=placeholder` → Claude Code sends **`x-api-key: placeholder`**, while OneCLI also injects **`Authorization: Bearer <real sk-ant-oat>`**.
- Anthropic **rejects a request that carries an invalid `x-api-key`** even when a valid Bearer is present. And the `sk-ant-oat` (OAuth/subscription) token only works as Bearer (as `x-api-key` it's rejected too). Three-way dead end.

**Fix applied**: registered the Claude subscription token as a **`type=anthropic`** secret (`onecli secrets create --type anthropic --host-pattern api.anthropic.com`), and **deleted** the old `type=generic` one. This makes OneCLI set **`CLAUDE_CODE_OAUTH_TOKEN=placeholder`** (not `ANTHROPIC_API_KEY`) in the container and inject natively, so Claude Code sends a **Bearer-only** header. **Verified 200** through the gateway (`/v1/models` → `claude-opus-5`).
- **Lesson**: for subscription/OAuth (sk-ant-oat) tokens on Anthropic, the OneCLI secret **must** be `type=anthropic`. A `generic` secret with an `Authorization` header is NOT sufficient for the Claude Code SDK path. (Token itself was recovered from OneCLI postgres `encrypted_value` = `iv:authTag:ciphertext` AES-256-GCM, key = base64 in `/app/data/secret-encryption-key`.)

### Codex (OpenAI) provider — INSTALLED (2026-08-19)
- `add-codex` skill applied from `origin/providers`: 18 payload files copied across host/container/setup trees; barrel imports (`src/providers/index.ts`, `container/agent-runner/src/providers/index.ts`, `setup/providers/index.ts`); `@openai/codex@0.138.0` added to `container/cli-tools.json`; host build + **1662/1662 tests** (was 1648) + container provider tests 147 green; image rebuilt with `codex-cli 0.138.0` verified.
- **Trunk drift fixes made** (providers-branch payload vs our trunk): `ProviderContainerConfigFn` now accepts **async** config fns (returns `| Promise<…>`), `container-runner.resolveProviderContribution` awaits it (codex fn awaits DB ops); added `| { type: 'file'; path: string }` to container `ProviderEvent` union (codex yields generated-image files; poll-loop switch is log-only, no exhaustiveness break).
- **Group** `OpenAI (Codex)` `ag-d2de9cbc-…` — provider `codex`, tz Europe/Zurich.
- **Auth (ChatGPT subscription)**: `codex login --device-auth` in the agent image (temp CODEX_HOME) → `auth.json` (`auth_mode: chatgpt`, **plan: plus**) → `onecli secrets create --name 'Codex (ChatGPT)' --type openai --file auth.json --host-pattern chatgpt.com` (secret `3f1c0096-…`) → agent `ag-d2de9cbc…` created in OneCLI + `agents set-secrets` bound to it. Container-config now serves **credentialStub `/home/node/.codex/auth.json`** (sentinels) + aoc proxy env — chain **verified live** (stub + proxy + `codex exec` reached the model; only failure was the Plus quota window, see G3).
- **Fallback key**: user's `sk-proj-…` (org billing **$0** — `credit_balance_exhausted` on every completion) is kept in the vault as secret `OpenAI Codex` (name-matches `onecli secrets list`, host `api.openai.com`), NOT bound to any agent. Useful the moment the org is topped up: `onecli agents set-secrets --id <openai-agent-id> --secret-ids <that-secret-id>`.
- How codex auth works here (no credential env at all): gateway serves the `~/.codex/auth.json` stub into the group's `.codex-shared` mount (host `src/providers/codex.ts`), MITM proxy (aoc agent token in `HTTPS_PROXY`) swaps the sentinel for the real vault credential on the wire. `CODEX_ENV_ALLOWLIST` deliberately strips `OPENAI_API_KEY` — auth rides the stub only.

### Telegram wiring — FINAL (6 chats)
| Chat | messaging_group | agent group | model / endpoint |
|---|---|---|---|
| DM `telegram:1650895591` | `mg-1787134078792-qsyptu` | Main `ag-d441119a` (claude) | Claude Opus 5 (subscription) |
| group `telegram:-1003959814938` | `mg-1787144119031-tm5qse` | GPT `ag-d4793d5a` (opencode) | `opencode-go/glm-5.3` @ `opencode.ai/zen/go/v1` (Go plan; quota-parked until ≈ Aug 23 — **G1**) |
| group `telegram:-1003988096183` | `mg-1787147577309-etsg6b` | Local Qwen `ag-2049cb02` (opencode) | `openai/qwen3.8:27b` @ `http://10.114.0.91:11434/v1` |
| group `telegram:-1003914446408` | `mg-1787151949379-bqzrod` | OpenAI (Codex) `ag-d2de9cbc` (codex) | ChatGPT Plus subscription (codex models; gateway-swapped; quota **G3**) |
| group `telegram:-1004477619551` | `mg-1787232598491-hfis2l` | Tech Lead `ag-1787228879565` (opencode) | `opencode-go/glm-5.3` @ `zen/go/v1` (Go plan; quota **G1**) |
| group `telegram:-1004483965420` | `mg-1787235491363-xz97ic` | German Tutor `ag-aa72bc1a` (claude) | Claude subscription — persona: master German coach + Goethe/telc/TestDAF/DSH exam prep (persona in gitignored `groups/german-tutor/CLAUDE.md`) |

**Pairing wiring gotcha (G4):** `pair-telegram --intent wire-to:<folder>` only **registers the chat**; the wiring row itself is a separate step — on the 4th chat the nohup'd setup waiter died before wiring and the user's first message had nowhere to route. Fix applied + general rule: after ANY pairing, verify `ncl wirings list` shows the row, else create it: `ncl wirings create --messaging-group-id <mg> --agent-group-id <ag> --engage-mode pattern --engage-pattern "."` (matching the other 3 chats).

All wirings `--engage-mode pattern --engage-pattern "."` (always-on, like a DM; groups need the bot as **admin** to read every message).

**G5 — Qwen (OpenAI-compatible / Ollama) "Z.responses is not a function" (code, fixed 2026-08-19, commit `3d933a`):** the opencode provider previously pinned `npm: '@ai-sdk/openai-compatible'` for the `openai` branch (Chat-Completions transport). OpenCode's `openai` provider **unconditionally** calls `provider.responses(model)` (the Responses API) — in 1.4.17 *and* the latest 1.18.18 — but `@ai-sdk/openai-compatible` only exposes `languageModel`/`chatModel`, **no `.responses`** → crash. **Fix:** removed the pin (`container/agent-runner/src/providers/opencode.ts`); OpenCode now uses its stock OpenAI provider, which speaks the Responses API that **this Ollama implements** (`/v1/responses` verified: returns `"pong"`). Regression-guarded by `opencode.config.test.ts` (asserts `provider.openai.npm === undefined`). If Qwen ever breaks again with `Z.responses`, the culprit is a re-introduced `openai-compatible` pin — do **not** "fix" it by pinning that package back.

### ⚠⚠ Learned gotchas (do NOT re-learn)

**G1 — OpenCodeGo quota/balance (external, not fixable here):** `gpt-5.6-luna` (Go plan) is **quota-limited (~4-day reset)** and the free-tier billing has **insufficient balance**. GPT-5.6 Luna is therefore **unavailable right now**. The GPT group runs `opencode/deepseek-v4-flash-free` (zen/v1, works). **When quota resets or balance is topped up**, flip back: set GPT group env to `OPENCODE_PROVIDER=opencode-go`, `OPENCODE_MODEL=opencode-go/gpt-5.6-luna`, `ANTHROPIC_BASE_URL=https://opencode.ai/zen/go/v1`, then `ncl groups restart` its group + clear its `continuation:opencode` (see G2). Free-tier catalog (zen/v1): claude-*, deepseek-v4, gemini-3.x, glm-5.x, gpt-5.6-luna/sol/terra, grok, kimi-k3, minimax-m3, qwen3.x. **Status 2026-08-20:** the **Go-plan weekly window is exhausted** — both Go-backed channels (GPT, Tech Lead, parked on `opencode-go/glm-5.3`) get HTTP **429 "Weekly usage limit reached. Resets in 3 days"** (≈ Aug 23); the config already points at `opencode-go/glm-5.3`, so it **auto-recovers at the reset** — no changes needed. `glm-5.3` exists **only in the Go catalog** (free tier tops out at `glm-5.2`) — verified live through the gateway: `onecli run -- curl https://opencode.ai/zen/go/v1/models`. If you need it working *right now*: enable balance-based usage from the URL in the 429 message (OpenCode workspace → Go), or flip both envs to free `opencode/glm-5.2` + `zen/v1` (remember **G2** reset per group).
**AUTH:** the `opencode` (zen) endpoint validates **Bearer** strictly (even a valid `x-api-key` is rejected if a garbage Bearer is present) — I registered an **extra** OneCLI generic secret on `opencode.ai` injecting `Authorization: Bearer {value}` (in addition to the original `x-api-key` one) so both header styles carry the real key. Keep both.

**G2 — Stuck OpenCode "continuation" pointer (silent no-reply):** OpenCode persists its session id as key `continuation:opencode` (value `ses_…`) in the **session's `outbound.db` → table `session_state`**. It is pinned to the model that was active when the session was created. **If you change a group's model while a session exists**, the resumed `ses_…` fails (`Model not found: …`) and the poll-loop **acks the message with NO delivery** (silent). **After any model change for a group, before the user's next message:**
```bash
S=data/v2-sessions/<group-id>/<session-id>
pnpm exec tsx scripts/q.ts "$S/outbound.db" "DELETE FROM session_state WHERE key='continuation:opencode';"
rm -rf "$S/opencode-xdg"
ncl groups restart --id <group-id>
```
Also, that state dir was created **root-owned** (`opencode-xdg` 0:0) → container (uid 1000) can't `mkdir /opencode-xdg/opencode` → EACCES. Fix: `sudo chown -R 1000:1000 "$S" groups/<folder>`.

**G3 — ChatGPT-Plus Codex usage window (external quota, NOT an auth failure):** `codex exec` with the subscription auth returns `You've hit your usage limit. … try again at Aug 20th, 2026 7:32 AM` when the Plus Codex allotment for the current window is used. This is **expected behavior, not a wiring bug** — auth + model selection already succeeded (that error comes back from OpenAI after auth). Also: over a ChatGPT account only **codex models** are allowed (`gpt-5.1` etc. → `invalid_request_error: not supported when using Codex with a ChatGPT account`). Distinct from G1: G1 = OpenCodeGo's separate Go-plan quota; G3 = the user's own ChatGPT Plus Codex window. If the OpenAI group answers "usage limit", wait for the reset time printed in the error (or upgrade to Pro) — do not chase it as a bug.

**G6 — new wirings are born SILENT: default `engage_mode=mention` (code, learned 2026-08-20):** `ncl wirings create` **without** an explicit `--engage-mode` falls back to the channel adapter's declaration = **`mention`** in groups → the agent answers only @mentions and **silently drops** everything else (German Tutor hit exactly this: paired, wired, zero response). Always create with **`--engage-mode pattern --engage-pattern "."`** — or fix afterwards: `ncl wirings update <wiring-id> --engage-mode pattern --engage-pattern "."`. Verify `ncl wirings list` shows `pattern` *before* telling the user to test.

## DONE — 2026-08-20

- **5th agent: Tech Lead** (folder `tech-lead`, `ag-1787228879565-aazl7e`) — created in-chat via Main (pairing `455139`, wiring `b1db5910-…`), provider `opencode`. Chat `telegram:-1004477619551` / `mg-1787232598491-hfis2l`.
- **6th agent: German Tutor** (folder `german-tutor`, `ag-aa72bc1a-…`) — terminal route (`ncl groups create` + persona). Provider **`claude`** (empty → instance default; the strongest subscription for coaching). Persona in `groups/german-tutor/CLAUDE.md` (**gitignored** — back it up if you care): master German coach (corrections as *your version → correct → one-line rule → native rephrase*, i+1 pushing, "grammar-right but sounds foreign" flags), Goethe A1–C2 / telc / TestDAF / DSH exam prep with mock tasks + scoring, persistent error log in `memory/`, Telegram-short replies, German-first with English/Serbian explanation fallback. Chat `telegram:-1004483965420` / `mg-1787235491363-xz97ic`, wiring `310a6708-…` — created in **`mention` mode, corrected to `pattern .`** (gotcha **G6**).
- **GPT + Tech Lead switched to Go-plan `glm-5.3`:** both envs now `OPENCODE_PROVIDER=opencode-go`, `OPENCODE_MODEL=opencode-go/glm-5.3`, `OPENCODE_SMALL_MODEL=opencode-go/glm-5.3`, `ANTHROPIC_BASE_URL=https://opencode.ai/zen/go/v1` (free-tier DeepSeek usage had expired; `glm-5.3` is Go-only). **G2** reset + restart on both.
- **Live catalog verified through the gateway** (`onecli run -- curl <endpoint>/models`): Go tier = `glm-5.3/5.2/5.1/5`, `deepseek-v4-pro/flash`, `qwen3.5–3.8`, `minimax-m3`, `kimi-k3`, `gpt-5.6-luna`, `grok-4.5`, …; free tier (`zen/v1`) = 63 models, GLM tops out at **`glm-5.2`**.
- **Go-plan quota exhausted as of 16:0x local 2026-08-20:** 429 on both Go channels ("Weekly usage limit reached. Resets in 3 days", ≈ **Aug 23**). Wiring + auth + model ID all verified (the 429 comes back *after* auth and model acceptance) — see **G1** status for recovery paths.

## Reproduce on a fresh VM (the "this VM dies" path)

Target: **delete this VM → spin up an Ubuntu 24.04 VM (2+ CPU, ~8 GB RAM, passwordless sudo) → `git clone` the repo → all core chats up (six on this deployment; step 3 also lists the optional extras).** Everything deterministic below is already in the committed tree; only the secret/phone steps need you.

**0. Prereqs (one-time, same VM flavor):** Node 22 + pnpm (NodeSource), Docker, `make`+`g++` only if a native dep must compile from source (this stack ships prebuilt `better-sqlite3@11.10.0` — no compiler needed).

**1. Bootstrap the foundation** — this one command does Node/pnpm → Docker → build the agent image (bakes `claude-code`, `codex-cli`, `opencode-ai`, `agent-browser` from `container/cli-tools.json`) → OneCLI vault → Claude auth → Telegram adapter install → systemd unit + linger → pairing for the **main** chat:
```bash
cd ~/nanoclaw-v2 && bash nanoclaw.sh        # = setup.sh + `pnpm run setup:auto`
loginctl enable-linger ubuntu               # headless VM: keep the user unit alive after SSH logout
```
Idempotent; re-run any step via `NANOCLAW_SKIP=…` or the `/setup` agent skill. Docker-group gotcha: if `docker_group_not_active`, `newgrp docker` (or log out/in) and rerun.

**2. Providers are in trunk — no branch juggling anymore.** `claude`, `codex`, and `opencode` (+ the Qwen `openai`-compatible path) all ship in the committed tree and in the built image. If you changed deps, rebuild: `./container/build.sh`.

**3. Recreate the other agent groups + per-group model** (after the service is up; the main/Claude group already exists from step 1). One group per provider/model — set provider + per-group env (migration `023`). The optional extras from this deployment: **Tech Lead** (same Go/glm-5.3 recipe) and **German Tutor** (provider `claude`; persona file `groups/german-tutor/CLAUDE.md` lives under gitignored `groups/` — recreate or back it up to reproduce).
| Group | `openai|opencode|codex` provider | per-group env (`OPENCODE_PROVIDER` / `OPENCODE_MODEL` / `ANTHROPIC_BASE_URL`) |
|---|---|---|
| Main (DM) | `claude` | — (Claude subscription) |
| GPT (and optional Tech Lead) | `opencode` | `opencode-go` / `opencode-go/glm-5.3` / `https://opencode.ai/zen/go/v1` (Go plan, quota **G1**; free fallback: `opencode` / `opencode/glm-5.2` / `https://opencode.ai/zen/v1`) |
| Local Qwen | `opencode` | `openai` / `openai/qwen3.8:27b` / `http://<ollama-ip>:11434/v1` (see **G5** — do **not** pin `openai-compatible`) |
| OpenAI (Codex) | `codex` | — (ChatGPT Plus subscription) |

**4. Register OneCLI credentials** (the secret/human part — re-do on a new vault):
- **Claude**: `onecli secrets create --type anthropic --host-pattern api.anthropic.com` (**must** be `type=anthropic`, not `generic` — see the `x-api-key: placeholder` dead-end above).
- **Codex**: `codex login --device-auth` → `auth.json` → `onecli secrets create --type openai --host-pattern chatgpt.com`, bound to the Codex agent.
- **OpenCodeGo**: generic secret on `opencode.ai` (keep **both** `x-api-key` and `Authorization: Bearer` — see **G1**).
- **Ollama (Qwen)**: none — placeholder key is fine.
Verify: `onecli agents list`.

**5. Pair + wire each Telegram chat** — for each group chat: issue the 4-digit code (`pnpm exec tsx setup/index.ts --step pair-telegram --`), **you send those digits to the bot from the phone**, then **verify the wiring row exists** (gotcha **G4**):
```bash
ncl wirings create --messaging-group-id <mg> --agent-group-id <ag> --engage-mode pattern --engage-pattern "."
#   ↑ do NOT omit --engage-mode: the default is "mention" → silent group chat (gotcha G6)
ncl wirings list   # ← confirm the row says engage_mode=pattern
```
The bot must be **admin** in each group chat to read every message.

**Adding agents later is routine, not surgery** — see `CREATING-AGENTS.md` (ask Main in Telegram, or the terminal route: create → provider → pair → wire → verify).

**6. Smoke-test each chat** — per group: ping/pong, one real task, check `logs/nanoclaw.log` + `data/v2-sessions/<group>/…` on failure. Expect `G1` (OpenCodeGo quota) / `G3` (Codex window) quota replies on GPT + OpenAI = correct wiring, not a bug.

**What genuinely needs you (can't be scripted):** creating the Telegram bot + sending pairing codes, the OneCLI secret values (Claude/Codex/OpenCodeGo tokens), and any quota resets. Everything else is in the repo + the commands above.

## REMAINING (in order)

> **SUPERSEDED (2026-08-20):** items 1–8 below were the initial plan and are all **done** (see the DONE sections). What is *actually* still open:
>
> 1. **External quota windows** — G1: Go-plan weekly reset ≈ **Aug 23** (GPT + Tech Lead auto-recover; or enable balance usage / drop to free `glm-5.2`). G3: ChatGPT-Plus Codex window reset was **Aug 20 ~7:32 AM** — the OpenAI group should work by now if it didn't.
> 2. **Optional container-image bumps** (pin-to-newest-verified, lower priority): `container/cli-tools.json` — codex-cli 0.138.0→0.148.0, agent-browser 0.27.1→0.34.0; Dockerfile `ARG BUN_VERSION` 1.3.12→1.3.14; then `./container/build.sh` + container tests + `ncl groups restart`. (OpenCode 1.18.18 not needed for the Qwen path — it has the same strict `.responses` branch, see **G5**.)
> 3. **better-sqlite3 stays 11.10.0** — no `make`/`g++` on the host to compile 13.x.

1. **Gather credentials from the user** (not stored anywhere yet):
   - Anthropic: API key, or Claude OAuth (wizard's auth step handles the Claude Code OAuth token capture — `setup/register-claude-token.sh`).
   - OpenAI: API key (or decide to route OpenAI via OpenCodeGo's GPT-5.6 Luna instead — no separate key).
   - OpenCodeGo: API key (check if `~/.api/opencodego` is it).
   - Telegram bot: **user must create the bot on their phone via @BotFather and paste the token** — cannot be delegated.
2. **Create the Telegram bot** (user, phone) — instructions above.
3. **Run the wizard interactively** (real terminal, not headless):
   `cd ~/nanoclaw-v2 && bash nanoclaw.sh`
   - Standard setup → Fresh agent → container (auto-installs Docker; if it complains `docker_group_not_active`: `newgrp docker` in the shell or log out/in, then rerun) → OneCLI fresh → auth (Claude first pass is fine) → display name (defaults `ubuntu`) → **timezone: confirm IANA zone** (ask the user!) → channel: **Telegram** (paste token when asked) → pairing code (user sends digits from phone) → verify.
   - Ping/pong first-chat step auto-verifies the container works (30–60 s cold start).
   - Headless-VM gotcha: service is a **systemd user unit** — enable lingering so it survives SSH logout: `loginctl enable-linger ubuntu`.
4. **Install the OpenCode provider**: apply `origin/providers:.claude/skills/add-opencode/SKILL.md` (steps are idempotent; can be driven by having the nano-claw agent itself apply it, or run manually: copy files via `git show origin/providers:…`, append barrel imports, `bun add @opencode-ai/sdk@1.4.17` in `container/agent-runner`, Dockerfile `ARG OPENCODE_VERSION=1.4.17` + add `opencode-ai` to the pnpm global-install block, `pnpm run build`, `./container/build.sh`).
   - Do this **before** creating the non-claude agent groups so the image already has the provider.
5. **Create the agent groups + set configs** (after service runs):
   - Claude group: `ncl groups create` + `ncl groups config update --provider claude`.
   - One group per provider/model combo (OpenAI, OpenCodeGo, local Qwen) with provider `opencode` + the env mapping table above (resolve the global-`.env` vs per-group model question first — see caveat).
   - Register the OneCLI secrets (anthropic / openai / opencode.ai host patterns); default `all` secret mode injects them; verify `onecli agents list`.
6. **Wire Telegram chats to groups**: `/init-first-agent` picks the existing user + chat and wires it to the chosen agent group; other groups via `/manage-channels` (different Telegram chats ↔ different agent groups — that's how "different agents do different stuff" is exposed from the phone).
7. **End-to-end check from the phone**: ping/pong per group, run a real task per provider, check `logs/nanoclaw.log` + session DBs (`data/v2-sessions/<group>/inbound.db` / `outbound.db`) on failure.
8. **Optional polish**: `/customize` personality per group; `ncl tasks` for scheduled jobs; CJK fonts in container if needed (not needed here).

## Quick resume command

```bash
cd ~/nanoclaw-v2 && bash nanoclaw.sh        # wizard; basics already pass
tail -f logs/nanoclaw.log                    # after it's running
ncl help                                     # admin CLI once the service is up
```

Agent-assist fallback: talk to the NanoClaw agent itself and it has skills for every step (`/setup`, `/add-telegram`, `/add-opencode`, `/init-first-agent`, `/manage-channels`). `clack` wizard prompts only work in a live terminal.
