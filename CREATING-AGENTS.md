# Creating agents in chat

How to add a **new agent** (a new persona with its own model, memory, and container)
and hook it to a Telegram chat — or point an existing chat at a different agent.

> In NanoClaw, an **agent** = one *agent group*: its own `groups/<folder>/` (persona
> `CLAUDE.md` + memory), its own container, its own model/subscription. Chats are wired
> to agents one-to-one; nothing crosses between agents unless you explicitly share.

## What you can use as backends

| Backend | Model(s) | Subscription / quota |
|---|---|---|
| **Claude** | Claude Opus / Sonnet (subscription) | your Claude plan |
| **OpenAI (Codex)** | codex models via ChatGPT account | ChatGPT Plus window (gotcha **G3**) |
| **OpenCodeGo** | deepseek-v4-flash-free, gpt-5.6-luna, qwen3.x, gemini-3.x, kimi-k3, glm-5.x, … | free tier works; Go-plan models are quota-gated (gotcha **G1**) |
| **Local Qwen** | `qwen3.8:27b` via Ollama | local compute — no quota |

Any agent can use any of these — the provider is set **per agent group**, so you can have
a Claude researcher, a Codex coder, and a Qwen summarizer side by side.

## Route 1 — Just ask an agent in Telegram (easiest)

Message **Main** (your DM — the owner agent, which can create agents; the group agents
are scope-limited to their own group) with a request like:

```text
Create a new agent for me.
Persona: "Investor Research" — it briefs me on the markets every morning,
concise, cites sources.
Run it on OpenAI/Codex.
```

Then walk the short flow it drives:

1. It **creates the agent group** (new persona folder + provider + per-group env).
2. It tells you **which Telegram group/chat** it should live in and hands you a
   **4-digit pairing code** (for a brand-new chat).
3. **You send that code** to the bot, from that chat (the bot must be an **admin**
   of the group; Group Privacy off for DMs).
4. It **wires** the chat ↔ agent (always-on by default; say the word for
   mention-only instead) and greets you to confirm.

That's the whole loop. It also works for *"the GPT chat should now be served by a
new local-Qwen agent called X"* — i.e. re-pointing an existing chat is the same
conversation.

## Route 2 — Terminal (deterministic / scriptable)

```bash
cd ~/nanoclaw-v2

# 1. create the agent group  (--folder <slug> is the on-disk identity, unique)
ncl groups list --json                          # see existing ones
ncl groups create --folder investor-research --name "Investor Research"

# 2. choose the provider (per-agent)
ncl groups config update --id <new-ag> --provider claude     # or codex, or opencode
# opencode backends additionally want per-group env, e.g.:
#   OPENCODE_PROVIDER=opencode
#   OPENCODE_MODEL=opencode/deepseek-v4-flash-free
#   ANTHROPIC_BASE_URL=https://opencode.ai/zen/v1

# 3. (new Telegram chat only) pair it
pnpm exec tsx setup/index.ts --step pair-telegram --
#   → prints a 4-digit CODE. Send that code to the bot from the new chat, then check:
ncl messaging-groups list --json                 # note the new mg-<id>

# 4. wire chat ↔ agent
ncl wirings create --messaging-group-id <mg> --agent-group-id <ag> \
    --engage-mode pattern --engage-pattern "."
#   engage-pattern:  "."  = respond to every message   (bot must be admin)
#                    "@bot" = mention-only
```

To **change the agent** of an existing wired chat instead:
`ncl wirings update --id <wiring> --agent-group-id <other-ag>` (or simply ask Main).

## Verify (always, after any create/wire)

```bash
ncl wirings list            # the row must exist ← gotcha G4: pairing ≠ wiring
tail -f logs/nanoclaw.log   # send a test message; watch route + container wake
```

A chat with no wiring row silently drops messages (this happened once — see gotcha
**G4** in `SETUP-STATUS.md`). If the agent doesn't answer: `data/v2-sessions/<ag>/<sess>/`
DBs (did inbound land? did outbound produce?) + `logs/nanoclaw.error.log`.

## Notes

- **Names/folders**: `groups/<folder>/` is created at group-create; the folder is the
  agent's identity on disk (persona + memory). Rename early, not late.
- **One provider per agent** — but any model *within* that provider (pick the model in
  the env/persona; switching models mid-session has the gotcha **G2** reset for opencode).
- **Quotas are per-subscription**: OpenCodeGo and ChatGPT-Plus windows don't share,
  and Claude/Ollama are independent. A new agent doesn't dilute the old ones.
- **Agent templates** (optional, upstream feature): `ncl groups create --template <ref>`
  stamps persona + MCP + skills from a bundle — see `docs/templates.md`.
