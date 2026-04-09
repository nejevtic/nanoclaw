/**
 * NanoClaw Agent Runner
 * Runs inside a container, receives config via stdin, outputs result to stdout
 *
 * Supports multi-backend dispatch: Anthropic (Claude Agent SDK via OneCLI),
 * Gemini (OpenAI-compatible), and Ollama. Backend selected via ACTIVE_BACKEND
 * secret. Anthropic uses the upstream SDK path with OneCLI credential proxy;
 * Ollama and Gemini use the direct API tool-call loop.
 *
 * Input protocol:
 *   Stdin: Full ContainerInput JSON (read until EOF)
 *   IPC:   Follow-up messages written as JSON files to /workspace/ipc/input/
 *          Files: {type:"message", text:"..."}.json — polled and consumed
 *          Sentinel: /workspace/ipc/input/_close — signals session end
 *
 * Stdout protocol:
 *   Each result is wrapped in OUTPUT_START_MARKER / OUTPUT_END_MARKER pairs.
 *   Multiple results may be emitted (one per agent teams result).
 */

import { execSync, spawn } from 'child_process';
import fs from 'fs';
import path from 'path';
import { execFile } from 'child_process';
import {
  query,
  HookCallback,
  PreCompactHookInput,
} from '@anthropic-ai/claude-agent-sdk';
import { fileURLToPath } from 'url';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface ContainerInput {
  prompt: string;
  sessionId?: string;
  groupFolder: string;
  chatJid: string;
  isMain: boolean;
  isScheduledTask?: boolean;
  assistantName?: string;
  script?: string;
  secrets?: Record<string, string>;
}

interface ContainerOutput {
  status: 'success' | 'error';
  result: string | null;
  newSessionId?: string;
  error?: string;
}

// Upstream types (for SDK path)
interface SessionEntry {
  sessionId: string;
  fullPath: string;
  summary: string;
  firstPrompt: string;
}

interface SessionsIndex {
  entries: SessionEntry[];
}

interface SDKUserMessage {
  type: 'user';
  message: { role: 'user'; content: string };
  parent_tool_use_id: null;
  session_id: string;
}

// Local types (for Ollama/Gemini path)
type Role = 'system' | 'user' | 'assistant' | 'tool';

interface SystemMessage    { role: 'system';    content: string }
interface UserMessage      { role: 'user';      content: string }
interface AssistantMessage { role: 'assistant'; content: string | null; tool_calls?: ToolCall[] }
interface ToolMessage      { role: 'tool';      tool_call_id: string; content: string }

type ChatMessage = SystemMessage | UserMessage | AssistantMessage | ToolMessage;

interface ToolCall {
  id: string;
  type: 'function';
  function: { name: string; arguments: string };
}

interface OllamaResponse {
  choices: Array<{
    message: {
      role: string;
      content: string | null;
      tool_calls?: ToolCall[];
    };
    finish_reason: string;
  }>;
}

interface HistoryFile {
  sessionId: string;
  messages: ChatMessage[];
}

interface BackendConfig {
  type: 'ollama' | 'anthropic' | 'gemini';
  ollamaBaseUrl: string;
  ollamaModel: string;
  oauthToken: string;
  geminiApiKey: string;
  geminiModel: string;
}

interface ParsedMessage {
  role: 'user' | 'assistant';
  content: string;
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const IPC_INPUT_DIR            = '/workspace/ipc/input';
const IPC_INPUT_CLOSE_SENTINEL = path.join(IPC_INPUT_DIR, '_close');
const IPC_MESSAGES_DIR         = '/workspace/ipc/messages';
const IPC_TASKS_DIR            = '/workspace/ipc/tasks';
const HISTORY_FILE             = '/workspace/group/.ollama-history.json';

const IPC_POLL_MS              = 500;
const MAX_TOOL_ITERATIONS      = 30;
const MAX_HISTORY_MESSAGES     = 60;   // non-system messages kept across turns
const TOOL_OUTPUT_MAX_CHARS    = 8000; // truncate large tool results

const OUTPUT_START_MARKER = '---NANOCLAW_OUTPUT_START---';
const OUTPUT_END_MARKER   = '---NANOCLAW_OUTPUT_END---';

// ---------------------------------------------------------------------------
// Logging / output helpers
// ---------------------------------------------------------------------------

function log(msg: string): void {
  console.error(`[agent-runner] ${msg}`);
}

function writeOutput(out: ContainerOutput): void {
  console.log(OUTPUT_START_MARKER);
  console.log(JSON.stringify(out));
  console.log(OUTPUT_END_MARKER);
}

// ---------------------------------------------------------------------------
// IPC helpers (shared by all backend paths)
// ---------------------------------------------------------------------------

function writeIpcFile(dir: string, data: object): void {
  fs.mkdirSync(dir, { recursive: true });
  const name = `${Date.now()}-${Math.random().toString(36).slice(2, 8)}.json`;
  const tmp  = path.join(dir, `${name}.tmp`);
  fs.writeFileSync(tmp, JSON.stringify(data, null, 2));
  fs.renameSync(tmp, path.join(dir, name));
}

function shouldClose(): boolean {
  if (fs.existsSync(IPC_INPUT_CLOSE_SENTINEL)) {
    try { fs.unlinkSync(IPC_INPUT_CLOSE_SENTINEL); } catch { /* ignore */ }
    return true;
  }
  return false;
}

function drainIpcInput(): string[] {
  try {
    fs.mkdirSync(IPC_INPUT_DIR, { recursive: true });
    const files = fs
      .readdirSync(IPC_INPUT_DIR)
      .filter((f) => f.endsWith('.json'))
      .sort();

    const messages: string[] = [];
    for (const file of files) {
      const filePath = path.join(IPC_INPUT_DIR, file);
      try {
        const data = JSON.parse(fs.readFileSync(filePath, 'utf-8'));
        fs.unlinkSync(filePath);
        if (data.type === 'message' && data.text) {
          messages.push(data.text);
        }
      } catch (err) {
        log(`Failed to process input file ${file}: ${err instanceof Error ? err.message : String(err)}`);
        try { fs.unlinkSync(filePath); } catch { /* ignore */ }
      }
    }
    return messages;
  } catch (err) {
    log(`IPC drain error: ${err instanceof Error ? err.message : String(err)}`);
    return [];
  }
}

function waitForIpcMessage(): Promise<string | null> {
  return new Promise((resolve) => {
    const poll = () => {
      if (shouldClose()) { resolve(null); return; }
      const messages = drainIpcInput();
      if (messages.length > 0) { resolve(messages.join('\n')); return; }
      setTimeout(poll, IPC_POLL_MS);
    };
    poll();
  });
}

// ---------------------------------------------------------------------------
// MessageStream — push-based async iterable for SDK path
// ---------------------------------------------------------------------------

/**
 * Push-based async iterable for streaming user messages to the SDK.
 * Keeps the iterable alive until end() is called, preventing isSingleUserTurn.
 */
class MessageStream {
  private queue: SDKUserMessage[] = [];
  private waiting: (() => void) | null = null;
  private done = false;

  push(text: string): void {
    this.queue.push({
      type: 'user',
      message: { role: 'user', content: text },
      parent_tool_use_id: null,
      session_id: '',
    });
    this.waiting?.();
  }

  end(): void {
    this.done = true;
    this.waiting?.();
  }

  async *[Symbol.asyncIterator](): AsyncGenerator<SDKUserMessage> {
    while (true) {
      while (this.queue.length > 0) {
        yield this.queue.shift()!;
      }
      if (this.done) return;
      await new Promise<void>((r) => {
        this.waiting = r;
      });
      this.waiting = null;
    }
  }
}

// ---------------------------------------------------------------------------
// Stdin reader
// ---------------------------------------------------------------------------

function readStdin(): Promise<string> {
  return new Promise((resolve, reject) => {
    let data = '';
    process.stdin.setEncoding('utf8');
    process.stdin.on('data', (chunk) => { data += chunk; });
    process.stdin.on('end', () => resolve(data));
    process.stdin.on('error', reject);
  });
}

// ---------------------------------------------------------------------------
// Tool definitions (OpenAI format, for Ollama/Gemini path)
// ---------------------------------------------------------------------------

function buildTools(isMain: boolean) {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const tools: any[] = [
    {
      type: 'function' as const,
      function: {
        name: 'bash',
        description: 'Run a bash command. cwd is /workspace/group. Avoid commands that run indefinitely.',
        parameters: {
          type: 'object',
          properties: {
            command: { type: 'string', description: 'Bash command to execute' },
          },
          required: ['command'],
        },
      },
    },
    {
      type: 'function' as const,
      function: {
        name: 'read_file',
        description: 'Read the full contents of a file.',
        parameters: {
          type: 'object',
          properties: {
            path: { type: 'string', description: 'Absolute or relative file path' },
          },
          required: ['path'],
        },
      },
    },
    {
      type: 'function' as const,
      function: {
        name: 'write_file',
        description: 'Write (or overwrite) a file with the given content.',
        parameters: {
          type: 'object',
          properties: {
            path:    { type: 'string', description: 'File path' },
            content: { type: 'string', description: 'Content to write' },
          },
          required: ['path', 'content'],
        },
      },
    },
    {
      type: 'function' as const,
      function: {
        name: 'edit_file',
        description: 'Replace an exact string in a file with a new string.',
        parameters: {
          type: 'object',
          properties: {
            path:       { type: 'string', description: 'File path to edit' },
            old_string: { type: 'string', description: 'Exact string to find and replace' },
            new_string: { type: 'string', description: 'Replacement string' },
          },
          required: ['path', 'old_string', 'new_string'],
        },
      },
    },
    {
      type: 'function' as const,
      function: {
        name: 'glob',
        description: 'Find files matching a glob pattern.',
        parameters: {
          type: 'object',
          properties: {
            pattern:   { type: 'string', description: 'Glob pattern, e.g. "**/*.md"' },
            directory: { type: 'string', description: 'Base directory (default: /workspace/group)' },
          },
          required: ['pattern'],
        },
      },
    },
    {
      type: 'function' as const,
      function: {
        name: 'grep',
        description: 'Search file contents for a pattern.',
        parameters: {
          type: 'object',
          properties: {
            pattern:   { type: 'string', description: 'Regex or literal string to search for' },
            path:      { type: 'string', description: 'File or directory to search (default: /workspace/group)' },
            file_glob: { type: 'string', description: 'Only search files matching this glob, e.g. "*.ts"' },
          },
          required: ['pattern'],
        },
      },
    },
    {
      type: 'function' as const,
      function: {
        name: 'send_message',
        description: "Send a message to the chat immediately while still running. Use for progress updates or multi-part replies. In Telegram, set sender to show a named bot identity.",
        parameters: {
          type: 'object',
          properties: {
            text:   { type: 'string', description: 'Message text to send' },
            sender: { type: 'string', description: 'Optional role/identity name shown in Telegram (e.g. "Researcher")' },
          },
          required: ['text'],
        },
      },
    },
    {
      type: 'function' as const,
      function: {
        name: 'schedule_task',
        description: 'Schedule a recurring or one-time task that runs as an agent.',
        parameters: {
          type: 'object',
          properties: {
            prompt:         { type: 'string', description: 'What the agent should do when the task runs' },
            schedule_type:  { type: 'string', enum: ['cron', 'interval', 'once'], description: 'cron=at specific times, interval=every N ms, once=single run' },
            schedule_value: { type: 'string', description: 'Cron expression, milliseconds, or local timestamp like "2026-02-01T15:30:00"' },
            context_mode:   { type: 'string', enum: ['group', 'isolated'], description: 'group=run with chat history, isolated=fresh session' },
          },
          required: ['prompt', 'schedule_type', 'schedule_value'],
        },
      },
    },
    {
      type: 'function' as const,
      function: {
        name: 'list_tasks',
        description: 'List all scheduled tasks.',
        parameters: { type: 'object', properties: {} },
      },
    },
    {
      type: 'function' as const,
      function: {
        name: 'cancel_task',
        description: 'Cancel and delete a scheduled task.',
        parameters: {
          type: 'object',
          properties: {
            task_id: { type: 'string', description: 'Task ID to cancel' },
          },
          required: ['task_id'],
        },
      },
    },
  ];

  if (isMain) {
    tools.push({
      type: 'function' as const,
      function: {
        name: 'register_group',
        description: 'Register a new chat group so the agent responds to it. Main group only. Use available_groups.json to find the JID.',
        parameters: {
          type: 'object',
          properties: {
            jid:     { type: 'string', description: 'Group JID (e.g. "tg:-1234567890" for Telegram)' },
            name:    { type: 'string', description: 'Display name' },
            folder:  { type: 'string', description: 'Folder name (lowercase, hyphens)' },
            trigger: { type: 'string', description: 'Trigger word (e.g. "@Andy")' },
          },
          required: ['jid', 'name', 'folder', 'trigger'],
        },
      },
    });
  }

  return tools;
}

// ---------------------------------------------------------------------------
// Tool execution (Ollama/Gemini path)
// ---------------------------------------------------------------------------

interface ToolCtx {
  chatJid: string;
  groupFolder: string;
  isMain: boolean;
}

function truncate(s: string): string {
  if (s.length <= TOOL_OUTPUT_MAX_CHARS) return s;
  return s.slice(0, TOOL_OUTPUT_MAX_CHARS) + `\n... (truncated, ${s.length} chars total)`;
}

async function executeTool(
  name: string,
  args: Record<string, string>,
  ctx: ToolCtx,
): Promise<string> {
  try {
    switch (name) {
      case 'bash': {
        try {
          const out = execSync(args.command, {
            encoding: 'utf-8',
            timeout: 60_000,
            maxBuffer: 2 * 1024 * 1024,
            cwd: '/workspace/group',
            shell: '/bin/bash',
          });
          return truncate(out || '(no output)');
        } catch (err: any) {
          const msg = [err.stdout, err.stderr, err.message].filter(Boolean).join('\n');
          return truncate(`Exit ${err.status ?? 1}:\n${msg}`);
        }
      }

      case 'read_file': {
        if (!fs.existsSync(args.path)) return `File not found: ${args.path}`;
        return truncate(fs.readFileSync(args.path, 'utf-8'));
      }

      case 'write_file': {
        fs.mkdirSync(path.dirname(args.path), { recursive: true });
        fs.writeFileSync(args.path, args.content, 'utf-8');
        return `Wrote ${args.content.length} bytes to ${args.path}`;
      }

      case 'edit_file': {
        if (!fs.existsSync(args.path)) return `File not found: ${args.path}`;
        const src = fs.readFileSync(args.path, 'utf-8');
        if (!src.includes(args.old_string)) return `String not found in ${args.path}:\n${args.old_string.slice(0, 120)}`;
        fs.writeFileSync(args.path, src.replace(args.old_string, args.new_string), 'utf-8');
        return 'File updated.';
      }

      case 'glob': {
        const dir = args.directory || '/workspace/group';
        try {
          const out = execSync(
            `find "${dir}" -path "${args.pattern.replace(/\*\*/g, '*')}" 2>/dev/null | head -200`,
            { encoding: 'utf-8', timeout: 10_000 },
          );
          return out.trim() || '(no files found)';
        } catch { return '(no files found)'; }
      }

      case 'grep': {
        const target    = args.path || '/workspace/group';
        const includeFlag = args.file_glob ? `--include="${args.file_glob}"` : '';
        try {
          const out = execSync(
            `grep -r ${includeFlag} -n "${args.pattern}" "${target}" 2>/dev/null | head -200`,
            { encoding: 'utf-8', timeout: 10_000 },
          );
          return truncate(out.trim() || '(no matches)');
        } catch { return '(no matches)'; }
      }

      case 'send_message': {
        writeIpcFile(IPC_MESSAGES_DIR, {
          type: 'message',
          chatJid: ctx.chatJid,
          text: args.text,
          sender: args.sender || undefined,
          groupFolder: ctx.groupFolder,
          timestamp: new Date().toISOString(),
        });
        return 'Message sent.';
      }

      case 'schedule_task': {
        writeIpcFile(IPC_TASKS_DIR, {
          type: 'schedule_task',
          prompt: args.prompt,
          schedule_type: args.schedule_type,
          schedule_value: args.schedule_value,
          context_mode: args.context_mode || 'group',
          targetJid: ctx.chatJid,
          createdBy: ctx.groupFolder,
          timestamp: new Date().toISOString(),
        });
        return `Task scheduled (${args.schedule_type}: ${args.schedule_value}).`;
      }

      case 'list_tasks': {
        const file = '/workspace/ipc/current_tasks.json';
        if (!fs.existsSync(file)) return 'No scheduled tasks.';
        const all = JSON.parse(fs.readFileSync(file, 'utf-8')) as Array<{
          id: string; prompt: string; schedule_type: string;
          schedule_value: string; status: string; groupFolder: string;
        }>;
        const tasks = ctx.isMain ? all : all.filter(t => t.groupFolder === ctx.groupFolder);
        if (tasks.length === 0) return 'No scheduled tasks.';
        return tasks.map(t =>
          `[${t.id}] ${t.prompt.slice(0, 60)}... | ${t.schedule_type}: ${t.schedule_value} | ${t.status}`
        ).join('\n');
      }

      case 'cancel_task': {
        writeIpcFile(IPC_TASKS_DIR, {
          type: 'cancel_task',
          taskId: args.task_id,
          groupFolder: ctx.groupFolder,
          isMain: ctx.isMain,
          timestamp: new Date().toISOString(),
        });
        return `Task ${args.task_id} cancelled.`;
      }

      case 'register_group': {
        if (!ctx.isMain) return 'Only the main group can register new groups.';
        writeIpcFile(IPC_TASKS_DIR, {
          type: 'register_group',
          jid: args.jid,
          name: args.name,
          folder: args.folder,
          trigger: args.trigger,
          timestamp: new Date().toISOString(),
        });
        return `Group "${args.name}" registered.`;
      }

      default:
        return `Unknown tool: ${name}`;
    }
  } catch (err) {
    return `Tool error: ${err instanceof Error ? err.message : String(err)}`;
  }
}

// ---------------------------------------------------------------------------
// Conversation history (Ollama/Gemini path)
// ---------------------------------------------------------------------------

function loadHistory(sessionId?: string): { messages: ChatMessage[]; sessionId: string } {
  const newId = sessionId || crypto.randomUUID();
  if (!sessionId || !fs.existsSync(HISTORY_FILE)) {
    return { messages: [], sessionId: newId };
  }
  try {
    const data: HistoryFile = JSON.parse(fs.readFileSync(HISTORY_FILE, 'utf-8'));
    if (data.sessionId !== sessionId) return { messages: [], sessionId: newId };
    const nonSystem = data.messages.filter(m => m.role !== 'system');
    return { messages: nonSystem.slice(-MAX_HISTORY_MESSAGES), sessionId: data.sessionId };
  } catch {
    return { messages: [], sessionId: newId };
  }
}

function saveHistory(sessionId: string, messages: ChatMessage[]): void {
  try {
    const nonSystem = messages.filter(m => m.role !== 'system').slice(-MAX_HISTORY_MESSAGES);
    fs.writeFileSync(HISTORY_FILE, JSON.stringify({ sessionId, messages: nonSystem }, null, 2));
  } catch (err) {
    log(`Failed to save history: ${err}`);
  }
}

// ---------------------------------------------------------------------------
// System prompt (Ollama/Gemini path)
// ---------------------------------------------------------------------------

function buildSystemPrompt(input: ContainerInput): string {
  const name = input.assistantName || 'Andy';
  const now  = new Date().toLocaleString('en-US', { dateStyle: 'full', timeStyle: 'short' });

  const claudeMdContent = ['/workspace/group/CLAUDE.md', '/workspace/global/CLAUDE.md']
    .filter(f => fs.existsSync(f))
    .map(f => { try { return '\n\n---\n' + fs.readFileSync(f, 'utf-8'); } catch { return ''; } })
    .join('');

  const toolList = [
    'bash', 'read_file', 'write_file', 'edit_file', 'glob', 'grep',
    'send_message', 'schedule_task', 'list_tasks', 'cancel_task',
    ...(input.isMain ? ['register_group'] : []),
  ].join(', ');

  return `You are ${name}, a helpful AI assistant running inside a secure container.
Current date/time: ${now}

You have access to the following tools: ${toolList}.

Guidelines:
- Use tools proactively to complete tasks fully before responding.
- Use bash for anything requiring computation, file listing, or system interaction.
- Use send_message to send progress updates mid-task for long operations.
- Keep final responses concise and direct.
- Wrap internal reasoning in <internal>...</internal> tags — it will be stripped before sending.
- For memory, write notes to /workspace/group/CLAUDE.md.${claudeMdContent}`;
}

// ---------------------------------------------------------------------------
// Ollama API call
// ---------------------------------------------------------------------------

async function callOllama(
  baseUrl: string,
  model: string,
  messages: ChatMessage[],
  tools: ReturnType<typeof buildTools>,
): Promise<OllamaResponse> {
  const res = await fetch(`${baseUrl}/v1/chat/completions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model, messages, tools, stream: false }),
  });

  if (!res.ok) {
    const body = await res.text();
    throw new Error(`Ollama ${res.status}: ${body.slice(0, 400)}`);
  }

  return res.json() as Promise<OllamaResponse>;
}

// ---------------------------------------------------------------------------
// Gemini (OpenAI-compatible endpoint)
// ---------------------------------------------------------------------------

async function callGemini(
  apiKey: string,
  model: string,
  messages: ChatMessage[],
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  tools: any[],
): Promise<OllamaResponse> {
  const res = await fetch(
    'https://generativelanguage.googleapis.com/v1beta/openai/chat/completions',
    {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${apiKey}`,
      },
      body: JSON.stringify({ model, messages, tools, stream: false }),
    },
  );

  if (!res.ok) {
    const body = await res.text();
    throw new Error(`Gemini ${res.status}: ${body.slice(0, 400)}`);
  }

  return res.json() as Promise<OllamaResponse>;
}

// ---------------------------------------------------------------------------
// Anthropic — Claude CLI subprocess (--print mode, uses CLAUDE_CODE_OAUTH_TOKEN)
// ---------------------------------------------------------------------------

const CLAUDE_CLI = '/app/node_modules/@anthropic-ai/claude-code/cli.js';

async function spawnClaude(
  args: string[],
  oauthToken: string,
): Promise<{ result: string | null; sessionId: string | undefined }> {
  return new Promise((resolve, reject) => {
    const env: NodeJS.ProcessEnv = {
      ...process.env,
      // Clear API key so the CLI uses OAuth (not paid API credits)
      ANTHROPIC_API_KEY: '',
      ...(oauthToken ? { CLAUDE_CODE_OAUTH_TOKEN: oauthToken } : {}),
    };

    const proc = spawn('node', [CLAUDE_CLI, ...args], {
      cwd: '/workspace/group',
      env,
      stdio: ['ignore', 'pipe', 'pipe'],
    });

    let stdout = '';
    proc.stdout.on('data', (d: Buffer) => { stdout += d.toString(); });
    proc.stderr.on('data', (d: Buffer) => {
      const line = d.toString().trim();
      if (line) log(`[claude-cli] ${line.slice(0, 200)}`);
    });

    proc.on('error', reject);
    proc.on('close', () => {
      try {
        const parsed = JSON.parse(stdout.trim());
        // Accept success or treat is_error=false as success
        if (parsed.subtype === 'success' || parsed.is_error === false) {
          resolve({
            result: parsed.result ?? null,
            sessionId: parsed.session_id ?? parsed.sessionId,
          });
        } else {
          reject(new Error(`Claude CLI: ${parsed.subtype ?? 'unknown'} — ${(parsed.result ?? '').slice(0, 200)}`));
        }
      } catch (err) {
        reject(new Error(`Claude CLI parse error: ${err} | stdout: ${stdout.slice(0, 300)}`));
      }
    });
  });
}

async function runQueryWithClaudeSDKCLI(
  prompt: string,
  sessionId: string | undefined,
  oauthToken: string,
  input: ContainerInput,
): Promise<{ newSessionId: string; closedDuringQuery: boolean; pendingIpc: string[] }> {
  let closedDuringQuery = false;
  const ipcBuffer: string[] = [];

  // IPC polling — set flag if close sentinel appears
  let ipcDone = false;
  const pollIpc = () => {
    if (ipcDone) return;
    if (shouldClose()) { closedDuringQuery = true; return; }
    ipcBuffer.push(...drainIpcInput());
    setTimeout(pollIpc, IPC_POLL_MS);
  };
  setTimeout(pollIpc, IPC_POLL_MS);

  const now = new Date().toLocaleString('en-US', { dateStyle: 'full', timeStyle: 'short' });
  const baseArgs = [
    '--print',
    '--output-format', 'json',
    '--dangerously-skip-permissions',
    '--append-system-prompt', `Current date/time: ${now}`,
    '-p', prompt,
  ];

  try {
    // Try resuming existing session; on failure fall back to new session
    let run: { result: string | null; sessionId: string | undefined };
    if (sessionId) {
      try {
        run = await spawnClaude([...baseArgs, '--resume', sessionId], oauthToken);
      } catch (err) {
        log(`Resume failed (${err}), starting fresh session`);
        run = await spawnClaude(baseArgs, oauthToken);
      }
    } else {
      run = await spawnClaude(baseArgs, oauthToken);
    }

    const newSessionId = run.sessionId ?? crypto.randomUUID();
    writeOutput({ status: 'success', result: run.result, newSessionId });
    return { newSessionId, closedDuringQuery, pendingIpc: [...ipcBuffer, ...drainIpcInput()] };
  } finally {
    ipcDone = true;
  }
}

// ---------------------------------------------------------------------------
// runQueryOllamaGemini: tool-call loop for Ollama and Gemini backends
// ---------------------------------------------------------------------------

async function runQueryOllamaGemini(
  prompt: string,
  sessionId: string | undefined,
  backend: BackendConfig,
  input: ContainerInput,
): Promise<{ newSessionId: string; closedDuringQuery: boolean; pendingIpc: string[] }> {
  const { messages: history, sessionId: activeSessionId } = loadHistory(sessionId);
  const tools  = buildTools(input.isMain);
  const system = buildSystemPrompt(input);
  const ctx: ToolCtx = { chatJid: input.chatJid, groupFolder: input.groupFolder, isMain: input.isMain };

  // Full message list for this turn
  const messages: ChatMessage[] = [
    { role: 'system', content: system },
    ...history,
    { role: 'user', content: prompt },
  ];

  // Poll IPC while the LLM is running
  let ipcDone = false;
  let closedDuringQuery = false;
  const ipcBuffer: string[] = [];

  const pollIpc = () => {
    if (ipcDone) return;
    if (shouldClose()) { closedDuringQuery = true; ipcDone = true; return; }
    ipcBuffer.push(...drainIpcInput());
    setTimeout(pollIpc, IPC_POLL_MS);
  };
  setTimeout(pollIpc, IPC_POLL_MS);

  let finalText: string | null = null;

  try {
    for (let i = 0; i < MAX_TOOL_ITERATIONS; i++) {
      log(`LLM call ${i + 1} [${backend.type}] (${messages.length} messages)...`);
      const res = backend.type === 'gemini'
        ? await callGemini(backend.geminiApiKey, backend.geminiModel, messages, tools)
        : await callOllama(backend.ollamaBaseUrl, backend.ollamaModel, messages, tools);
      const choice = res.choices[0];
      if (!choice) throw new Error('Backend returned no choices');

      const msg = choice.message;
      messages.push({ role: 'assistant', content: msg.content ?? null, tool_calls: msg.tool_calls });

      if (msg.tool_calls && msg.tool_calls.length > 0) {
        for (const tc of msg.tool_calls) {
          let args: Record<string, string> = {};
          try { args = JSON.parse(tc.function.arguments); } catch { /* leave empty */ }
          log(`Tool: ${tc.function.name}(${JSON.stringify(args).slice(0, 120)})`);
          const result = await executeTool(tc.function.name, args, ctx);
          log(`Result: ${result.slice(0, 120)}`);
          messages.push({ role: 'tool', tool_call_id: tc.id, content: result });
        }
        // Continue loop to get next LLM response
      } else {
        // No tool calls → final answer
        finalText = msg.content ?? null;
        break;
      }
    }
  } finally {
    ipcDone = true;
  }

  // Save history (exclude system prompt)
  saveHistory(activeSessionId, messages.filter(m => m.role !== 'system'));

  // Emit result
  writeOutput({ status: 'success', result: finalText, newSessionId: activeSessionId });

  // Collect any IPC that arrived during the query
  const pendingIpc = [...ipcBuffer, ...drainIpcInput()];

  return { newSessionId: activeSessionId, closedDuringQuery, pendingIpc };
}

// ---------------------------------------------------------------------------
// Session summary helpers (SDK path)
// ---------------------------------------------------------------------------

function getSessionSummary(
  sessionId: string,
  transcriptPath: string,
): string | null {
  const projectDir = path.dirname(transcriptPath);
  const indexPath = path.join(projectDir, 'sessions-index.json');

  if (!fs.existsSync(indexPath)) {
    log(`Sessions index not found at ${indexPath}`);
    return null;
  }

  try {
    const index: SessionsIndex = JSON.parse(
      fs.readFileSync(indexPath, 'utf-8'),
    );
    const entry = index.entries.find((e) => e.sessionId === sessionId);
    if (entry?.summary) {
      return entry.summary;
    }
  } catch (err) {
    log(
      `Failed to read sessions index: ${err instanceof Error ? err.message : String(err)}`,
    );
  }

  return null;
}

/**
 * Archive the full transcript to conversations/ before compaction.
 */
function createPreCompactHook(assistantName?: string): HookCallback {
  return async (input, _toolUseId, _context) => {
    const preCompact = input as PreCompactHookInput;
    const transcriptPath = preCompact.transcript_path;
    const sessionId = preCompact.session_id;

    if (!transcriptPath || !fs.existsSync(transcriptPath)) {
      log('No transcript found for archiving');
      return {};
    }

    try {
      const content = fs.readFileSync(transcriptPath, 'utf-8');
      const messages = parseTranscript(content);

      if (messages.length === 0) {
        log('No messages to archive');
        return {};
      }

      const summary = getSessionSummary(sessionId, transcriptPath);
      const name = summary ? sanitizeFilename(summary) : generateFallbackName();

      const conversationsDir = '/workspace/group/conversations';
      fs.mkdirSync(conversationsDir, { recursive: true });

      const date = new Date().toISOString().split('T')[0];
      const filename = `${date}-${name}.md`;
      const filePath = path.join(conversationsDir, filename);

      const markdown = formatTranscriptMarkdown(
        messages,
        summary,
        assistantName,
      );
      fs.writeFileSync(filePath, markdown);

      log(`Archived conversation to ${filePath}`);
    } catch (err) {
      log(
        `Failed to archive transcript: ${err instanceof Error ? err.message : String(err)}`,
      );
    }

    return {};
  };
}

function sanitizeFilename(summary: string): string {
  return summary
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, 50);
}

function generateFallbackName(): string {
  const time = new Date();
  return `conversation-${time.getHours().toString().padStart(2, '0')}${time.getMinutes().toString().padStart(2, '0')}`;
}

function parseTranscript(content: string): ParsedMessage[] {
  const messages: ParsedMessage[] = [];

  for (const line of content.split('\n')) {
    if (!line.trim()) continue;
    try {
      const entry = JSON.parse(line);
      if (entry.type === 'user' && entry.message?.content) {
        const text =
          typeof entry.message.content === 'string'
            ? entry.message.content
            : entry.message.content
                .map((c: { text?: string }) => c.text || '')
                .join('');
        if (text) messages.push({ role: 'user', content: text });
      } else if (entry.type === 'assistant' && entry.message?.content) {
        const textParts = entry.message.content
          .filter((c: { type: string }) => c.type === 'text')
          .map((c: { text: string }) => c.text);
        const text = textParts.join('');
        if (text) messages.push({ role: 'assistant', content: text });
      }
    } catch {}
  }

  return messages;
}

function formatTranscriptMarkdown(
  messages: ParsedMessage[],
  title?: string | null,
  assistantName?: string,
): string {
  const now = new Date();
  const formatDateTime = (d: Date) =>
    d.toLocaleString('en-US', {
      month: 'short',
      day: 'numeric',
      hour: 'numeric',
      minute: '2-digit',
      hour12: true,
    });

  const lines: string[] = [];
  lines.push(`# ${title || 'Conversation'}`);
  lines.push('');
  lines.push(`Archived: ${formatDateTime(now)}`);
  lines.push('');
  lines.push('---');
  lines.push('');

  for (const msg of messages) {
    const sender = msg.role === 'user' ? 'User' : assistantName || 'Assistant';
    const content =
      msg.content.length > 2000
        ? msg.content.slice(0, 2000) + '...'
        : msg.content;
    lines.push(`**${sender}**: ${content}`);
    lines.push('');
  }

  return lines.join('\n');
}

// ---------------------------------------------------------------------------
// runQuery (SDK path) — uses Claude Agent SDK with OneCLI credentials
// ---------------------------------------------------------------------------

/**
 * Run a single query using the Claude Agent SDK.
 * Uses MessageStream (AsyncIterable) to keep isSingleUserTurn=false,
 * allowing agent teams subagents to run to completion.
 * Also pipes IPC messages into the stream during the query.
 */
async function runQuery(
  prompt: string,
  sessionId: string | undefined,
  mcpServerPath: string,
  containerInput: ContainerInput,
  sdkEnv: Record<string, string | undefined>,
  resumeAt?: string,
): Promise<{
  newSessionId?: string;
  lastAssistantUuid?: string;
  closedDuringQuery: boolean;
}> {
  const stream = new MessageStream();
  stream.push(prompt);

  // Poll IPC for follow-up messages and _close sentinel during the query
  let ipcPolling = true;
  let closedDuringQuery = false;
  const pollIpcDuringQuery = () => {
    if (!ipcPolling) return;
    if (shouldClose()) {
      log('Close sentinel detected during query, ending stream');
      closedDuringQuery = true;
      stream.end();
      ipcPolling = false;
      return;
    }
    const messages = drainIpcInput();
    for (const text of messages) {
      log(`Piping IPC message into active query (${text.length} chars)`);
      stream.push(text);
    }
    setTimeout(pollIpcDuringQuery, IPC_POLL_MS);
  };
  setTimeout(pollIpcDuringQuery, IPC_POLL_MS);

  let newSessionId: string | undefined;
  let lastAssistantUuid: string | undefined;
  let messageCount = 0;
  let resultCount = 0;

  // Load global CLAUDE.md as additional system context (shared across all groups)
  const globalClaudeMdPath = '/workspace/global/CLAUDE.md';
  let globalClaudeMd: string | undefined;
  if (!containerInput.isMain && fs.existsSync(globalClaudeMdPath)) {
    globalClaudeMd = fs.readFileSync(globalClaudeMdPath, 'utf-8');
  }

  // Discover additional directories mounted at /workspace/extra/*
  // These are passed to the SDK so their CLAUDE.md files are loaded automatically
  const extraDirs: string[] = [];
  const extraBase = '/workspace/extra';
  if (fs.existsSync(extraBase)) {
    for (const entry of fs.readdirSync(extraBase)) {
      const fullPath = path.join(extraBase, entry);
      if (fs.statSync(fullPath).isDirectory()) {
        extraDirs.push(fullPath);
      }
    }
  }
  if (extraDirs.length > 0) {
    log(`Additional directories: ${extraDirs.join(', ')}`);
  }

  for await (const message of query({
    prompt: stream,
    options: {
      cwd: '/workspace/group',
      additionalDirectories: extraDirs.length > 0 ? extraDirs : undefined,
      resume: sessionId,
      resumeSessionAt: resumeAt,
      systemPrompt: globalClaudeMd
        ? {
            type: 'preset' as const,
            preset: 'claude_code' as const,
            append: globalClaudeMd,
          }
        : undefined,
      allowedTools: [
        'Bash',
        'Read',
        'Write',
        'Edit',
        'Glob',
        'Grep',
        'WebSearch',
        'WebFetch',
        'Task',
        'TaskOutput',
        'TaskStop',
        'TeamCreate',
        'TeamDelete',
        'SendMessage',
        'TodoWrite',
        'ToolSearch',
        'Skill',
        'NotebookEdit',
        'mcp__nanoclaw__*',
      ],
      env: sdkEnv,
      permissionMode: 'bypassPermissions',
      allowDangerouslySkipPermissions: true,
      settingSources: ['project', 'user'],
      mcpServers: {
        nanoclaw: {
          command: 'node',
          args: [mcpServerPath],
          env: {
            NANOCLAW_CHAT_JID: containerInput.chatJid,
            NANOCLAW_GROUP_FOLDER: containerInput.groupFolder,
            NANOCLAW_IS_MAIN: containerInput.isMain ? '1' : '0',
          },
        },
      },
      hooks: {
        PreCompact: [
          { hooks: [createPreCompactHook(containerInput.assistantName)] },
        ],
      },
    },
  })) {
    messageCount++;
    const msgType =
      message.type === 'system'
        ? `system/${(message as { subtype?: string }).subtype}`
        : message.type;
    log(`[msg #${messageCount}] type=${msgType}`);

    if (message.type === 'assistant' && 'uuid' in message) {
      lastAssistantUuid = (message as { uuid: string }).uuid;
    }

    if (message.type === 'system' && message.subtype === 'init') {
      newSessionId = message.session_id;
      log(`Session initialized: ${newSessionId}`);
    }

    if (
      message.type === 'system' &&
      (message as { subtype?: string }).subtype === 'task_notification'
    ) {
      const tn = message as {
        task_id: string;
        status: string;
        summary: string;
      };
      log(
        `Task notification: task=${tn.task_id} status=${tn.status} summary=${tn.summary}`,
      );
    }

    if (message.type === 'result') {
      resultCount++;
      const textResult =
        'result' in message ? (message as { result?: string }).result : null;
      log(
        `Result #${resultCount}: subtype=${message.subtype}${textResult ? ` text=${textResult.slice(0, 200)}` : ''}`,
      );
      writeOutput({
        status: 'success',
        result: textResult || null,
        newSessionId,
      });
    }
  }

  ipcPolling = false;
  log(
    `Query done. Messages: ${messageCount}, results: ${resultCount}, lastAssistantUuid: ${lastAssistantUuid || 'none'}, closedDuringQuery: ${closedDuringQuery}`,
  );
  return { newSessionId, lastAssistantUuid, closedDuringQuery };
}

// ---------------------------------------------------------------------------
// Script runner (for scheduled tasks with a pre-check script)
// ---------------------------------------------------------------------------

interface ScriptResult {
  wakeAgent: boolean;
  data?: unknown;
}

const SCRIPT_TIMEOUT_MS = 30_000;

async function runScript(script: string): Promise<ScriptResult | null> {
  const scriptPath = '/tmp/task-script.sh';
  fs.writeFileSync(scriptPath, script, { mode: 0o755 });

  return new Promise((resolve) => {
    execFile(
      'bash',
      [scriptPath],
      {
        timeout: SCRIPT_TIMEOUT_MS,
        maxBuffer: 1024 * 1024,
        env: process.env,
      },
      (error, stdout, stderr) => {
        if (stderr) {
          log(`Script stderr: ${stderr.slice(0, 500)}`);
        }

        if (error) {
          log(`Script error: ${error.message}`);
          return resolve(null);
        }

        // Parse last non-empty line of stdout as JSON
        const lines = stdout.trim().split('\n');
        const lastLine = lines[lines.length - 1];
        if (!lastLine) {
          log('Script produced no output');
          return resolve(null);
        }

        try {
          const result = JSON.parse(lastLine);
          if (typeof result.wakeAgent !== 'boolean') {
            log(
              `Script output missing wakeAgent boolean: ${lastLine.slice(0, 200)}`,
            );
            return resolve(null);
          }
          resolve(result as ScriptResult);
        } catch {
          log(`Script output is not valid JSON: ${lastLine.slice(0, 200)}`);
          resolve(null);
        }
      },
    );
  });
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

async function main(): Promise<void> {
  let input: ContainerInput;

  try {
    const raw = await readStdin();
    input = JSON.parse(raw);
    try { fs.unlinkSync('/tmp/input.json'); } catch { /* may not exist */ }
    log(`Input received for group: ${input.groupFolder}`);
  } catch (err) {
    writeOutput({
      status: 'error',
      result: null,
      error: `Failed to parse input: ${err instanceof Error ? err.message : String(err)}`,
    });
    process.exit(1);
  }

  // Multi-backend configuration: read active backend and credentials from secrets
  const secrets       = input.secrets || {};
  const activeBackend = secrets.ACTIVE_BACKEND === 'anthropic' ? 'anthropic'
                      : secrets.ACTIVE_BACKEND === 'gemini'    ? 'gemini'
                      : 'ollama';
  const ollamaBaseUrl = secrets.OLLAMA_BASE_URL || process.env.OLLAMA_BASE_URL || 'http://localhost:11434';
  const ollamaModel   = secrets.OLLAMA_MODEL    || process.env.OLLAMA_MODEL    || 'gpt-oss:latest';
  const oauthToken    = secrets.CLAUDE_CODE_OAUTH_TOKEN || process.env.CLAUDE_CODE_OAUTH_TOKEN || '';
  const geminiApiKey  = secrets.GEMINI_API_KEY  || process.env.GEMINI_API_KEY  || '';
  const geminiModel   = secrets.GEMINI_MODEL    || process.env.GEMINI_MODEL    || 'gemini-2.0-flash';

  // OneCLI credential proxy: credentials are injected via ANTHROPIC_BASE_URL.
  // CLAUDE_CODE_AUTO_COMPACT_WINDOW: 165k auto-compact threshold (upstream f77f9ce).
  const sdkEnv: Record<string, string | undefined> = {
    ...process.env,
    CLAUDE_CODE_AUTO_COMPACT_WINDOW: '165000',
  };

  // SDK path: mcpServerPath for nanoclaw MCP server
  const __dirname = path.dirname(fileURLToPath(import.meta.url));
  const mcpServerPath = path.join(__dirname, 'ipc-mcp-stdio.js');

  // Shared credentials for all backends
  const baseConfig = { ollamaBaseUrl, ollamaModel, oauthToken, geminiApiKey, geminiModel };

  // Fixed fallback priority: anthropic → gemini → ollama.
  // Chain starts from the configured active backend; remaining backends are tried in order.
  const FALLBACK_ORDER: Array<BackendConfig['type']> = ['anthropic', 'gemini', 'ollama'];
  const startIdx = FALLBACK_ORDER.indexOf(activeBackend as BackendConfig['type']);
  const orderedTypes = startIdx >= 0
    ? [...FALLBACK_ORDER.slice(startIdx), ...FALLBACK_ORDER.slice(0, startIdx)]
    : [...FALLBACK_ORDER];
  const fallbackChain: BackendConfig[] = orderedTypes.map(type => ({ ...baseConfig, type }));

  log(`Backend: ${activeBackend} | fallback chain: ${orderedTypes.join(' → ')}`);

  fs.mkdirSync(IPC_INPUT_DIR, { recursive: true });

  // Clean up stale _close sentinel from previous container runs
  try { fs.unlinkSync(IPC_INPUT_CLOSE_SENTINEL); } catch { /* ignore */ }

  // Build initial prompt (drain any stale IPC messages)
  let prompt = input.isScheduledTask
    ? `[SCHEDULED TASK]\n\n${input.prompt}`
    : input.prompt;

  const stale = drainIpcInput();
  if (stale.length > 0) prompt += '\n' + stale.join('\n');

  // Script phase: run script before waking agent
  if (input.script && input.isScheduledTask) {
    log('Running task script...');
    const scriptResult = await runScript(input.script);

    if (!scriptResult || !scriptResult.wakeAgent) {
      const reason = scriptResult ? 'wakeAgent=false' : 'script error/no output';
      log(`Script decided not to wake agent: ${reason}`);
      writeOutput({ status: 'success', result: null });
      return;
    }

    log(`Script wakeAgent=true, enriching prompt with data`);
    prompt = `[SCHEDULED TASK]\n\nScript output:\n${JSON.stringify(scriptResult.data, null, 2)}\n\nInstructions:\n${input.prompt}`;
  }

  let sessionId = input.sessionId;
  // Start from the full chain; updated each turn to prefer the last working backend
  let preferredChain = fallbackChain;

  // Query loop: run query → wait for IPC message → run new query → repeat
  let resumeAt: string | undefined;

  try {
    while (true) {
      log(`Query start (session: ${sessionId || 'new'})`);

      if (activeBackend === 'anthropic') {
        // Use Claude Agent SDK path (OneCLI credentials, full tool support)
        const queryResult = await runQuery(
          prompt,
          sessionId,
          mcpServerPath,
          input,
          sdkEnv,
          resumeAt,
        );
        if (queryResult.newSessionId) {
          sessionId = queryResult.newSessionId;
        }
        if (queryResult.lastAssistantUuid) {
          resumeAt = queryResult.lastAssistantUuid;
        }

        if (queryResult.closedDuringQuery) {
          log('Close sentinel consumed during query, exiting');
          break;
        }

        // Emit session update so host can track it
        writeOutput({ status: 'success', result: null, newSessionId: sessionId });

        log('Query ended, waiting for next IPC message...');
        const nextMessage = await waitForIpcMessage();
        if (nextMessage === null) {
          log('Close sentinel received, exiting');
          break;
        }
        log(`Got new message (${nextMessage.length} chars), starting new query`);
        prompt = nextMessage;
      } else {
        // Use Ollama/Gemini path: direct API call with tool-call loop and fallback chain
        let result: { newSessionId: string; closedDuringQuery: boolean; pendingIpc: string[] } | null = null;
        let lastErr: unknown;
        let succeededBackend: BackendConfig | null = null;

        for (const b of preferredChain) {
          if (b.type === 'anthropic') continue; // skip SDK backend in non-anthropic chain
          try {
            log(`Trying backend: ${b.type}`);
            result = await runQueryOllamaGemini(prompt, sessionId, b, input);
            succeededBackend = b;
            break;
          } catch (err) {
            const msg = err instanceof Error ? err.message : String(err);
            log(`Backend ${b.type} failed: ${msg.slice(0, 300)}, trying next backend...`);
            lastErr = err;
          }
        }

        if (!result || !succeededBackend) throw lastErr ?? new Error('All backends failed');

        // Drop any backends that failed this turn — don't retry them later in the session.
        const succeededIdx = preferredChain.indexOf(succeededBackend);
        preferredChain = preferredChain.slice(succeededIdx);

        sessionId = result.newSessionId;

        if (result.closedDuringQuery) {
          log('Closed during query, exiting');
          break;
        }

        // Emit session-update marker so host tracks the session
        writeOutput({ status: 'success', result: null, newSessionId: sessionId });

        // If IPC messages arrived during the query, use them as the next prompt
        if (result.pendingIpc.length > 0) {
          log(`${result.pendingIpc.length} IPC message(s) buffered during query, processing now`);
          prompt = result.pendingIpc.join('\n');
          continue;
        }

        log('Waiting for next IPC message...');
        const next = await waitForIpcMessage();
        if (next === null) {
          log('Close sentinel received, exiting');
          break;
        }

        log(`Next message received (${next.length} chars)`);
        prompt = next;
      }
    }
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    log(`Fatal error: ${msg}`);
    writeOutput({ status: 'error', result: null, newSessionId: sessionId, error: msg });
    process.exit(1);
  }
}

main();
