# NanoClaw Migration Guide

Generated: 2026-05-19
Base: 934f063aff5c30e7b49ce58b53b41901d3472a3e
HEAD at generation: 900c4e1 (style: apply prettier formatting)
Upstream: upstream/main

## Migration Plan

### Order of operations

1. Start from clean upstream/main checkout in worktree
2. Apply Telegram channel skill (`/add-telegram`) -- v2 has a native skill for this
3. Apply Emacs channel skill (if upstream skill exists)
4. Apply Ollama MCP tool skill (if upstream skill exists)
5. Re-implement multi-backend AI dispatch on v2's architecture
6. Re-implement credential proxy (or verify OneCLI covers the use case)
7. Re-implement OneCLI 2s timeout wrapper
8. Add OAuth token refresh script
9. Copy agent persona (CLAUDE.md files)
10. Opt out of telemetry
11. Validate build + tests

### Risk areas

- **Multi-backend dispatch**: v2's agent-runner is completely rewritten (Bun runtime, DB-based IPC, provider abstraction). The v1 agent-runner code cannot be ported directly -- it must be reimplemented using v2's `container/agent-runner/src/providers/` pattern.
- **Telegram channel**: v2 has its own `/add-telegram` skill on the `channels` branch with a Chat SDK bridge. The v1 grammy-based adapter must be replaced, but bot pool + `/backend` + `/project` commands need to be re-added on top.
- **Credential proxy**: v2 uses OneCLI natively. The custom credential proxy may be redundant, but the user wants to keep it as a fallback.

### Architecture changes (v1 -> v2)

| v1 Concept | v2 Equivalent |
|------------|---------------|
| `src/db.ts` (single SQLite) | `src/db/` directory + central DB (`data/v2.db`) + per-session DBs |
| `src/channels/registry.ts` | `src/channels/channel-registry.ts` + Chat SDK bridge |
| IPC via filesystem (JSON files) | Two-DB session split (inbound.db / outbound.db) |
| `container/agent-runner/src/index.ts` (monolith) | Provider abstraction: `container/agent-runner/src/providers/` |
| `getActiveBackend()` in `src/db.ts` | Provider config in `container_configs` table |
| `src/container-runner.ts` env injection | `src/container-config.ts` + `ncl groups config update` |
| Agent name in CLAUDE.md | Persona in agent group config |
| `registered_groups` table | `messaging_groups` + `agent_groups` + wirings |

---

## Applied Skills

Re-apply these by running the corresponding `/add-*` skill in the v2 worktree:

- **Telegram** -- run `/add-telegram` skill (v2 native, on `channels` branch)
- **Emacs** -- run `/add-emacs` skill (v2 native, on `channels` branch if available, or `upstream/skill/emacs`)
- **Ollama MCP tool** -- run `/add-ollama-tool` skill
- **Compact** -- built into v2 natively (no separate skill needed)
- **Channel formatting** -- built into v2 natively (no separate skill needed)

---

## Skill Interactions

None expected -- v2 skills are designed to be composable and install via the skill registry without conflicts.

---

## Modifications to Applied Skills

### Telegram: Bot pool + /backend + /project commands

**Intent:** After applying the base Telegram skill, add bot pool support (multiple bot tokens for agent teams), `/backend` command for runtime AI backend switching, and `/project` commands for project context injection.

**Files:** The v2 Telegram adapter (wherever `/add-telegram` installs it)

**How to apply:** After `/add-telegram` is applied, these features need to be re-implemented on top of v2's Telegram adapter. The v1 implementation details are in Section "Customizations > Telegram Bot Pool" below.

### OneCLI: 2-second timeout wrapper

**Intent:** Prevent container startup hangs when the OneCLI gateway is unresponsive. Falls back to `.env`-based credential injection.

**Files:** `src/container-runner.ts`

**How to apply:** After the base v2 is in place, find where `onecli.applyContainerConfig()` is called in `src/container-runner.ts` and wrap it:

```typescript
const onecliApplied = await Promise.race<boolean>([
  onecli.applyContainerConfig(args, {
    addHostMapping: false,
    agent: agentIdentifier,
  }),
  new Promise<boolean>((resolve) => setTimeout(() => resolve(false), 2000)),
]).catch(() => false);
if (onecliApplied) {
  logger.info({ containerName }, 'OneCLI gateway config applied');
} else {
  logger.warn(
    { containerName },
    'OneCLI gateway unreachable or timed out — falling back to .env credentials',
  );
}
```

---

## Customizations

### Multi-Backend AI Dispatch (Ollama / Gemini / OpenAI / Anthropic)

**Intent:** Support 4 AI backends with runtime hot-switching via Telegram `/backend` command, automatic fallback chain when a backend fails, and per-session backend persistence. The default backend is stored in the central DB and can be switched without restarting the service.

**Files (v2 equivalents):**
- `container/agent-runner/src/providers/` -- add new provider implementations
- `src/db/` -- store active backend preference
- `src/container-runner.ts` or `src/container-config.ts` -- inject backend env vars

**Architecture in v2:**

v2 already has a provider abstraction at `container/agent-runner/src/providers/`. The `claude` provider is built in; others (like `opencode`) are installed via skills. The multi-backend system should follow this pattern:

1. **Provider implementations** -- create `ollama.ts`, `gemini.ts`, `openai.ts` in `container/agent-runner/src/providers/`
2. **Backend selection** -- store active backend in `container_configs` table or a dedicated DB field
3. **Hot-switching** -- Telegram `/backend` command writes to DB; next container spawn reads it
4. **Fallback chain** -- `['anthropic', 'openai', 'gemini', 'ollama']`

**Key v1 code to preserve (for reference during reimplementation):**

The v1 multi-backend dispatch is a single 1,659-line file (`container/agent-runner/src/index.ts`). The critical sections are:

**Backend config and detection (v1 lines ~1370-1400):**
```typescript
const backendRaw = process.env.ACTIVE_BACKEND || secrets.ACTIVE_BACKEND;
let activeBackend: BackendConfig['type'] = backendRaw === 'anthropic' ? 'anthropic'
                    : backendRaw === 'gemini'    ? 'gemini'
                    : backendRaw === 'openai'    ? 'openai'
                    : 'ollama';
const ollamaBaseUrl = process.env.OLLAMA_BASE_URL || secrets.OLLAMA_BASE_URL || 'http://localhost:11434';
const ollamaModel   = process.env.OLLAMA_MODEL    || secrets.OLLAMA_MODEL    || 'gpt-oss:latest';
const oauthToken    = process.env.CLAUDE_CODE_OAUTH_TOKEN || secrets.CLAUDE_CODE_OAUTH_TOKEN || '';
const geminiApiKey  = process.env.GEMINI_API_KEY  || secrets.GEMINI_API_KEY  || '';
const geminiModel   = process.env.GEMINI_MODEL    || secrets.GEMINI_MODEL    || 'gemini-2.0-flash';
const openaiApiKey  = process.env.OPENAI_API_KEY  || secrets.OPENAI_API_KEY  || '';
const openaiModel   = process.env.OPENAI_MODEL    || secrets.OPENAI_MODEL    || 'gpt-5.5';
```

**Fallback chain logic (v1 lines ~1399-1410):**
```typescript
const FALLBACK_ORDER: Array<BackendConfig['type']> = ['anthropic', 'openai', 'gemini', 'ollama'];

function buildFallbackChain(primary: BackendConfig['type']): BackendConfig[] {
  const startIdx = FALLBACK_ORDER.indexOf(primary);
  const ordered = startIdx >= 0
    ? [...FALLBACK_ORDER.slice(startIdx), ...FALLBACK_ORDER.slice(0, startIdx)]
    : [...FALLBACK_ORDER];
  return ordered.map(type => ({ ...baseConfig, type }));
}
```

**Ollama API call (OpenAI-compatible):**
```typescript
async function callOllama(baseUrl: string, model: string, messages: ChatMessage[], tools: any[]): Promise<OllamaResponse> {
  const res = await fetch(`${baseUrl}/v1/chat/completions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model, messages, tools, stream: false }),
  });
  if (!res.ok) throw new Error(`Ollama ${res.status}: ${(await res.text()).slice(0, 400)}`);
  return res.json() as Promise<OllamaResponse>;
}
```

**Gemini API call (OpenAI-compatible endpoint):**
```typescript
async function callGemini(apiKey: string, model: string, messages: ChatMessage[], tools: any[]): Promise<OllamaResponse> {
  const res = await fetch('https://generativelanguage.googleapis.com/v1beta/openai/chat/completions', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${apiKey}` },
    body: JSON.stringify({ model, messages, tools, stream: false }),
  });
  if (!res.ok) throw new Error(`Gemini ${res.status}: ${(await res.text()).slice(0, 400)}`);
  return res.json() as Promise<OllamaResponse>;
}
```

**OpenAI API call:**
```typescript
async function callOpenAI(apiKey: string, model: string, messages: ChatMessage[], tools: any[]): Promise<OllamaResponse> {
  const res = await fetch('https://api.openai.com/v1/chat/completions', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${apiKey}` },
    body: JSON.stringify({ model, messages, tools, stream: false }),
  });
  if (!res.ok) throw new Error(`OpenAI ${res.status}: ${(await res.text()).slice(0, 400)}`);
  return res.json() as Promise<OllamaResponse>;
}
```

**Tool definitions (OpenAI format, for non-Anthropic backends):**
- `bash` -- run shell commands, 60s timeout, cwd /workspace/group
- `read_file`, `write_file`, `edit_file` -- filesystem ops
- `glob`, `grep` -- file search
- `send_message` -- sends to chat via IPC (in v2: via outbound.db)
- `schedule_task`, `list_tasks`, `cancel_task` -- task scheduling
- `register_group` -- main-only group registration

**Tool execution loop (v1 lines ~860-940):**
```typescript
for (let i = 0; i < MAX_TOOL_ITERATIONS; i++) {
  const res = backend.type === 'openai'
    ? await callOpenAI(backend.openaiApiKey, backend.openaiModel, messages, tools)
    : backend.type === 'gemini'
    ? await callGemini(backend.geminiApiKey, backend.geminiModel, messages, tools)
    : await callOllama(backend.ollamaBaseUrl, backend.ollamaModel, messages, tools);
  const choice = res.choices[0];
  const msg = choice.message;
  messages.push({ role: 'assistant', content: msg.content ?? null, tool_calls: msg.tool_calls });

  if (msg.tool_calls && msg.tool_calls.length > 0) {
    for (const tc of msg.tool_calls) {
      let args = {}; try { args = JSON.parse(tc.function.arguments); } catch {}
      const result = await executeTool(tc.function.name, args, ctx);
      messages.push({ role: 'tool', tool_call_id: tc.id, content: result });
    }
  } else {
    finalText = msg.content ?? null;
    break;
  }
}
```

**Conversation history persistence:**
- Stored at `/workspace/group/.ollama-history.json`
- Max 60 non-system messages
- Session-keyed (new session = fresh history)

**Host-side backend env var injection (`src/container-runner.ts`):**
```typescript
function readBackendEnvVars(): Record<string, string> {
  const vars = readEnvFile([
    'OLLAMA_BASE_URL', 'OLLAMA_MODEL',
    'GEMINI_API_KEY', 'GEMINI_MODEL',
    'CLAUDE_CODE_OAUTH_TOKEN',
    'OPENAI_API_KEY', 'OPENAI_MODEL',
  ]);
  vars.ACTIVE_BACKEND = getRouterState('active_backend') || 'ollama';
  return vars;
}
```

### Telegram Bot Pool (Agent Teams)

**Intent:** Multiple Telegram bot tokens for agent teams -- each sub-agent gets a distinct bot identity. Bots are assigned round-robin per sender and renamed on first use. Falls back to main bot when pool is empty.

**Files:** Telegram channel adapter

**How to apply:** After the Telegram channel is installed via `/add-telegram`, add bot pool support. The v2 Telegram adapter uses the Chat SDK bridge, so this may need to be added at a different layer.

Key code from v1:

```typescript
const poolApis: Api[] = [];
const senderBotMap = new Map<string, number>();
let nextPoolIndex = 0;
let mainBotApi: Api | null = null;

export async function initBotPool(tokens: string[]): Promise<void> {
  for (const token of tokens) {
    const api = new Api(token);
    const me = await api.getMe();
    poolApis.push(api);
  }
}

export async function sendPoolMessage(
  chatId: string, text: string, sender: string, groupFolder: string,
): Promise<void> {
  if (poolApis.length === 0) {
    // fallback to main bot with "*sender*: text" prefix
    return;
  }
  const key = `${groupFolder}:${sender}`;
  let idx = senderBotMap.get(key);
  if (idx === undefined) {
    idx = nextPoolIndex % poolApis.length;
    nextPoolIndex++;
    senderBotMap.set(key, idx);
    await poolApis[idx].setMyName(sender);
    await new Promise(r => setTimeout(r, 2000)); // Telegram name propagation delay
  }
  await poolApis[idx].sendMessage(chatId.replace(/^tg:/, ''), text);
}
```

Env var: `TELEGRAM_BOT_POOL` (comma-separated tokens in `.env`)

### Telegram /backend Command

**Intent:** Runtime AI backend switching from Telegram chat.

**How to apply:** Register a `/backend` command handler in the Telegram adapter:
- No arg: show current backend and usage
- `ollama`, `anthropic`, `gemini`, `openai`: write to DB, confirm in chat
- v2 equivalent of `setActiveBackend()`: write to `container_configs` or a router state table

### Telegram /project Command

**Intent:** Per-chat project context injection. Each project maps to a container mount and gets its CLAUDE.md prepended to prompts.

**How to apply:** Projects are defined in `workspace/projects.json`:
```json
{
  "projects": [
    {
      "name": "myproject",
      "displayName": "My Project",
      "containerPath": "/workspace/extra/myproject",
      "description": "Description here"
    }
  ]
}
```

Register one Telegram command per project (`/myproject`) plus `/project off`. Store active project per chat JID in the DB.

### Credential Proxy

**Intent:** HTTP proxy that intercepts container API calls and injects real credentials (API key or OAuth token). Containers never see raw secrets. Two auth modes: API key injection via `x-api-key` header, or OAuth token injection via `Authorization: Bearer` header.

**Files:** Create `src/credential-proxy.ts` (or adapt to v2's OneCLI pattern)

**How to apply:**

v2 uses OneCLI natively for credential injection. Check if OneCLI covers all the use cases. If the custom proxy is still needed as a fallback, the implementation is:

```typescript
import { createServer } from 'http';
import { request as httpsRequest } from 'https';

export function startCredentialProxy(port: number, host = '127.0.0.1'): Promise<Server> {
  const secrets = readEnvFile(['ANTHROPIC_API_KEY', 'CLAUDE_CODE_OAUTH_TOKEN', 'ANTHROPIC_AUTH_TOKEN', 'ANTHROPIC_BASE_URL']);
  const authMode = secrets.ANTHROPIC_API_KEY ? 'api-key' : 'oauth';
  const oauthToken = secrets.CLAUDE_CODE_OAUTH_TOKEN || secrets.ANTHROPIC_AUTH_TOKEN;
  const upstreamUrl = new URL(secrets.ANTHROPIC_BASE_URL || 'https://api.anthropic.com');

  return new Promise((resolve, reject) => {
    const server = createServer((req, res) => {
      const chunks: Buffer[] = [];
      req.on('data', c => chunks.push(c));
      req.on('end', () => {
        const body = Buffer.concat(chunks);
        const headers = { ...req.headers, host: upstreamUrl.host, 'content-length': body.length };
        delete headers['connection']; delete headers['keep-alive']; delete headers['transfer-encoding'];

        if (authMode === 'api-key') {
          delete headers['x-api-key'];
          headers['x-api-key'] = secrets.ANTHROPIC_API_KEY;
        } else if (headers['authorization']) {
          delete headers['authorization'];
          if (oauthToken) headers['authorization'] = `Bearer ${oauthToken}`;
        }

        const upstream = httpsRequest({ hostname: upstreamUrl.hostname, port: 443, path: req.url, method: req.method, headers }, upRes => {
          res.writeHead(upRes.statusCode!, upRes.headers);
          upRes.pipe(res);
        });
        upstream.on('error', err => { if (!res.headersSent) { res.writeHead(502); res.end('Bad Gateway'); } });
        upstream.write(body);
        upstream.end();
      });
    });
    server.listen(port, host, () => resolve(server));
    server.on('error', reject);
  });
}
```

### readEnvFile Utility

**Intent:** Parse `.env` without polluting `process.env`. Secrets stay out of the process environment so they don't leak to child processes.

**Files:** `src/env.ts` (or equivalent in v2)

**How to apply:**

v2 may already have this (check `src/env.ts`). If not:

```typescript
export function readEnvFile(keys: string[]): Record<string, string> {
  const envFile = path.join(process.cwd(), '.env');
  let content: string;
  try { content = fs.readFileSync(envFile, 'utf-8'); } catch { return {}; }

  const result: Record<string, string> = {};
  const wanted = new Set(keys);
  for (const line of content.split('\n')) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith('#')) continue;
    const eqIdx = trimmed.indexOf('=');
    if (eqIdx === -1) continue;
    const key = trimmed.slice(0, eqIdx).trim();
    if (!wanted.has(key)) continue;
    let value = trimmed.slice(eqIdx + 1).trim();
    if (value.length >= 2 && ((value.startsWith('"') && value.endsWith('"')) || (value.startsWith("'") && value.endsWith("'")))) {
      value = value.slice(1, -1);
    }
    if (value) result[key] = value;
  }
  return result;
}
```

### OAuth Token Refresh Script

**Intent:** Automatically refresh the Claude OAuth token before it expires. Reads refresh token from `~/.claude/.credentials.json`, calls the Anthropic OAuth endpoint, writes new access token back to credentials file and syncs to `.env`.

**Files:** `scripts/refresh-oauth-token.mjs`

**How to apply:** Copy the script as-is (it's standalone, no v1-specific dependencies):

```javascript
#!/usr/bin/env node
import fs from 'fs';
import path from 'path';
import os from 'os';

const CREDENTIALS_FILE = path.join(os.homedir(), '.claude', '.credentials.json');
const ENV_FILE = path.join(os.homedir(), 'nanoclaw', '.env');
const TOKEN_ENDPOINT = 'https://platform.claude.com/v1/oauth/token';
const CLIENT_ID = '9d1c250a-e61b-44d9-88ed-5944d1962f5e';
const REFRESH_THRESHOLD_MS = 60 * 60 * 1000;

async function main() {
  if (!fs.existsSync(CREDENTIALS_FILE)) { console.error('No credentials file'); process.exit(1); }
  const creds = JSON.parse(fs.readFileSync(CREDENTIALS_FILE, 'utf-8'));
  const oauth = creds?.claudeAiOauth;
  if (!oauth?.refreshToken) { console.error('No refresh token'); process.exit(1); }

  const remaining = (oauth.expiresAt ?? 0) - Date.now();
  if (remaining > REFRESH_THRESHOLD_MS) { console.log('Token still valid'); process.exit(0); }

  const body = new URLSearchParams({
    grant_type: 'refresh_token',
    refresh_token: oauth.refreshToken,
    client_id: CLIENT_ID,
  });

  const response = await fetch(TOKEN_ENDPOINT, {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body: body.toString(),
  });

  if (!response.ok) { console.error(`Token refresh failed: ${response.status}`); process.exit(1); }
  const result = JSON.parse(await response.text());

  oauth.accessToken = result.access_token;
  if (result.refresh_token) oauth.refreshToken = result.refresh_token;
  if (result.expires_in) oauth.expiresAt = Date.now() + result.expires_in * 1000;

  fs.writeFileSync(CREDENTIALS_FILE, JSON.stringify(creds, null, 2) + '\n');

  // Sync to .env
  if (fs.existsSync(ENV_FILE)) {
    const env = fs.readFileSync(ENV_FILE, 'utf-8');
    const updated = env.replace(/^CLAUDE_CODE_OAUTH_TOKEN=.*/m, `CLAUDE_CODE_OAUTH_TOKEN=${result.access_token}`);
    if (updated !== env) fs.writeFileSync(ENV_FILE, updated);
  }
}

main().catch(err => { console.error(err.message); process.exit(1); });
```

Set up systemd timer to run every 30 minutes:
```bash
# ~/.config/systemd/user/nanoclaw-token-refresh.service
[Unit]
Description=Refresh NanoClaw Claude OAuth token

[Service]
Type=oneshot
ExecStart=/usr/bin/node %h/nanoclaw/scripts/refresh-oauth-token.mjs

# ~/.config/systemd/user/nanoclaw-token-refresh.timer
[Unit]
Description=Refresh NanoClaw Claude OAuth token every 30 minutes

[Timer]
OnBootSec=5min
OnUnitActiveSec=30min

[Install]
WantedBy=timers.target
```

### Agent Persona (Andy)

**Intent:** Named agent "Andy" with specific capabilities, formatting rules, memory patterns, and scheduling guidance.

**Files:** `groups/main/CLAUDE.md`, `groups/global/CLAUDE.md`

**How to apply:** These are user content files in the `groups/` data directory. They are NOT touched during code migration -- they persist across upgrades. If the `groups/` directory is preserved (which it should be), no action needed.

If starting fresh, the persona is "Andy" with these key behaviors:
- Web search, browser automation, file I/O, bash, scheduling
- Internal thoughts in `<internal>` tags
- Channel-aware formatting (WhatsApp/Telegram: single `*bold*`, no `##` headings; Slack: mrkdwn; Discord: standard MD)
- Memory via files in workspace
- Task scripts with `wakeAgent` conditional logic to reduce API spend

### Telemetry Opt-Out

**Intent:** No outbound data from this box except the user's own Telegram channel. Opt out of all diagnostics permanently.

**Files:** `.claude/skills/setup/diagnostics.md`, `.claude/skills/update-nanoclaw/diagnostics.md`

**How to apply:** After upgrade, check if v2 has diagnostics sections in skills. If so, replace their content with:
```markdown
# Diagnostics -- opted out
```

Also remove any `## Diagnostics` sections from the corresponding SKILL.md files that reference these diagnostics files.

### Dependencies

**Host (`package.json`):**
- `grammy` ^1.39.3 -- Telegram bot framework (may be installed by `/add-telegram` skill)
- `pino` ^9.6.0, `pino-pretty` ^13.0.0 -- structured logging (check if v2 already uses pino)
- `yaml` ^2.8.2 -- YAML parsing
- `zod` ^4.3.6 -- schema validation (check if v2 already has zod)

**Container (`container/agent-runner/package.json`):**
- `@anthropic-ai/claude-code` -- Claude CLI for OAuth token refresh (check if v2 already includes this)

### .env Keys Required

```
ANTHROPIC_API_KEY=
CLAUDE_CODE_OAUTH_TOKEN=
TELEGRAM_BOT_TOKEN=
TELEGRAM_BOT_POOL=        # comma-separated pool bot tokens
OLLAMA_BASE_URL=
OLLAMA_MODEL=
GEMINI_API_KEY=
GEMINI_MODEL=
OPENAI_API_KEY=
OPENAI_MODEL=
ASSISTANT_NAME=
TELEGRAM_ONLY=            # optional, skip other channels
```
