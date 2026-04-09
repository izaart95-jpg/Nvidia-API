/**
 * NVIDIA Nemotron Proxy Server
 * ─────────────────────────────────────────────────────────────
 * OpenAI-compatible : POST /v1/chat/completions
 * Anthropic-compat  : POST /v1/messages
 * Health            : GET  /health
 * ─────────────────────────────────────────────────────────────
 * Env vars:
 *   NVIDIA_API_KEY   (required)
 *   PORT             (default 3000)
 */

import http from 'http';
import { URL } from 'url';

// ── Config ────────────────────────────────────────────────────
const PORT             = parseInt(process.env.PORT || '3000', 10);
const NVIDIA_API_KEY   = process.env.NVIDIA_API_KEY;
const NVIDIA_BASE      = 'https://integrate.api.nvidia.com/v1';
const DEFAULT_MODEL    = 'nvidia/nemotron-3-super-120b-a12b';
const DEFAULT_MAX_TOK  = 16384;

if (!NVIDIA_API_KEY) {
  console.error('[ERROR] NVIDIA_API_KEY env var is required');
  process.exit(1);
}

// ── Active client tracking ────────────────────────────────────
const activeClients = new Set();

// ── Utility: send JSON error ──────────────────────────────────
function sendError(res, status, message, type = 'proxy_error') {
  if (res.headersSent) return;
  res.writeHead(status, { 'Content-Type': 'application/json' });
  res.end(JSON.stringify({ error: { message, type, code: status } }));
}

// ── Utility: CORS headers ─────────────────────────────────────
function setCORSHeaders(res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization, x-api-key, anthropic-version');
}

// ── Parse request body ────────────────────────────────────────
function readBody(req) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    req.on('data', c => chunks.push(c));
    req.on('end', () => {
      try {
        resolve(JSON.parse(Buffer.concat(chunks).toString() || '{}'));
      } catch (e) {
        reject(new Error('Invalid JSON body'));
      }
    });
    req.on('error', reject);
  });
}

// ── Convert Anthropic → OpenAI messages format ────────────────
function anthropicToOpenAIMessages(body) {
  const messages = [];

  // system prompt
  if (body.system) {
    messages.push({ role: 'system', content: body.system });
  }

  // messages array
  for (const msg of (body.messages || [])) {
    if (typeof msg.content === 'string') {
      messages.push({ role: msg.role, content: msg.content });
    } else if (Array.isArray(msg.content)) {
      // content blocks → concatenate text parts
      const text = msg.content
        .filter(b => b.type === 'text')
        .map(b => b.text)
        .join('\n');
      messages.push({ role: msg.role, content: text });
    }
  }

  return messages;
}

// ── Build upstream request payload ───────────────────────────
function buildUpstreamPayload(messages, opts) {
  return {
    model:                opts.model || DEFAULT_MODEL,
    messages,
    max_tokens:           opts.max_tokens ?? DEFAULT_MAX_TOK,
    chat_template_kwargs: opts.chat_template_kwargs ?? { enable_thinking: true },
    stream:               opts.stream !== false, // default true
  };
}

// ── Convert OpenAI chunk → Anthropic SSE chunk ───────────────
function openAIChunkToAnthropic(chunk, index) {
  const choice = chunk.choices?.[0];
  if (!choice) return null;

  const delta = choice.delta || {};

  if (index === 0) {
    // message_start
    return [
      {
        type: 'message_start',
        message: {
          id: chunk.id,
          type: 'message',
          role: 'assistant',
          model: chunk.model,
          content: [],
          stop_reason: null,
          usage: { input_tokens: 0, output_tokens: 0 },
        },
      },
      { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } },
      { type: 'ping' },
    ];
  }

  if (choice.finish_reason) {
    return [
      { type: 'content_block_stop', index: 0 },
      {
        type: 'message_delta',
        delta: { stop_reason: choice.finish_reason === 'stop' ? 'end_turn' : choice.finish_reason },
        usage: { output_tokens: chunk.usage?.completion_tokens ?? 0 },
      },
      { type: 'message_stop' },
    ];
  }

  const text = delta.content || delta.reasoning_content || '';
  if (!text) return null;

  return [{ type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text } }];
}

// ── Convert OpenAI full response → Anthropic full response ───
function openAIToAnthropicFull(data) {
  const choice  = data.choices?.[0] || {};
  const content = choice.message?.content || '';
  return {
    id:           data.id,
    type:         'message',
    role:         'assistant',
    model:        data.model,
    content:      [{ type: 'text', text: content }],
    stop_reason:  choice.finish_reason === 'stop' ? 'end_turn' : choice.finish_reason,
    usage: {
      input_tokens:  data.usage?.prompt_tokens     ?? 0,
      output_tokens: data.usage?.completion_tokens ?? 0,
    },
  };
}

// ── Core: stream upstream → client (OpenAI passthrough) ──────
async function streamOpenAI(res, payload) {
  const upstreamRes = await fetchUpstream(payload);

  res.writeHead(200, {
    'Content-Type':      'text/event-stream',
    'Cache-Control':     'no-cache',
    'Connection':        'keep-alive',
    'X-Accel-Buffering': 'no',
  });

  const reader  = upstreamRes.body.getReader();
  const decoder = new TextDecoder();
  let   buffer  = '';

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop(); // keep incomplete line

      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed) continue;

        if (trimmed === 'data: [DONE]') {
          res.write('data: [DONE]\n\n');
          continue;
        }

        if (trimmed.startsWith('data: ')) {
          try {
            const chunk = JSON.parse(trimmed.slice(6));
            res.write(`data: ${JSON.stringify(chunk)}\n\n`);
          } catch {
            res.write(`${line}\n`);
          }
        }
      }
    }
    // flush remaining buffer
    if (buffer.trim()) res.write(`${buffer}\n`);
  } finally {
    reader.releaseLock();
    res.end();
  }
}

// ── Core: stream upstream → client (Anthropic SSE) ───────────
async function streamAnthropic(res, payload) {
  const upstreamRes = await fetchUpstream(payload);

  res.writeHead(200, {
    'Content-Type':      'text/event-stream',
    'Cache-Control':     'no-cache',
    'Connection':        'keep-alive',
    'X-Accel-Buffering': 'no',
  });

  const reader  = upstreamRes.body.getReader();
  const decoder = new TextDecoder();
  let   buffer  = '';
  let   index   = 0;

  function writeEvent(eventType, data) {
    res.write(`event: ${eventType}\ndata: ${JSON.stringify(data)}\n\n`);
  }

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop();

      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed || trimmed === 'data: [DONE]') continue;

        if (trimmed.startsWith('data: ')) {
          try {
            const chunk   = JSON.parse(trimmed.slice(6));
            const events  = openAIChunkToAnthropic(chunk, index++);
            if (!events) continue;
            for (const ev of events) {
              writeEvent(ev.type, ev);
            }
          } catch {
            /* skip malformed */
          }
        }
      }
    }
  } finally {
    reader.releaseLock();
    res.end();
  }
}

// ── Core: non-streaming (OpenAI) ─────────────────────────────
async function fullOpenAI(res, payload) {
  const upstreamRes = await fetchUpstream(payload);
  const data        = await upstreamRes.json();
  res.writeHead(200, { 'Content-Type': 'application/json' });
  res.end(JSON.stringify(data));
}

// ── Core: non-streaming (Anthropic) ──────────────────────────
async function fullAnthropic(res, payload) {
  const upstreamRes = await fetchUpstream(payload);
  const data        = await upstreamRes.json();
  res.writeHead(200, { 'Content-Type': 'application/json' });
  res.end(JSON.stringify(openAIToAnthropicFull(data)));
}

// ── Fetch upstream ────────────────────────────────────────────
async function fetchUpstream(payload) {
  const res = await fetch(`${NVIDIA_BASE}/chat/completions`, {
    method:  'POST',
    headers: {
      'Content-Type':  'application/json',
      'Authorization': `Bearer ${NVIDIA_API_KEY}`,
      'Accept':        payload.stream ? 'text/event-stream' : 'application/json',
    },
    body: JSON.stringify(payload),
  });

  if (!res.ok) {
    const text = await res.text();
    throw Object.assign(new Error(text), { status: res.status });
  }

  return res;
}

// ── Route: POST /v1/chat/completions  (OpenAI-compatible) ────
async function handleOpenAI(req, res) {
  const body    = await readBody(req);
  const stream  = body.stream !== false;
  const payload = buildUpstreamPayload(body.messages || [], {
    ...body,
    model:  body.model  || DEFAULT_MODEL,
    stream,
  });

  console.log(`[OpenAI]  model=${payload.model} stream=${stream} client=${req.socket.remoteAddress}`);

  if (stream) {
    await streamOpenAI(res, payload);
  } else {
    await fullOpenAI(res, payload);
  }
}

// ── Route: POST /v1/messages  (Anthropic-compatible) ─────────
async function handleAnthropic(req, res) {
  const body     = await readBody(req);
  const stream   = body.stream !== false;
  const messages = anthropicToOpenAIMessages(body);
  const payload  = buildUpstreamPayload(messages, {
    model:      body.model      || DEFAULT_MODEL,
    max_tokens: body.max_tokens || DEFAULT_MAX_TOK,
    stream,
  });

  console.log(`[Anthropic] model=${payload.model} stream=${stream} client=${req.socket.remoteAddress}`);

  if (stream) {
    await streamAnthropic(res, payload);
  } else {
    await fullAnthropic(res, payload);
  }
}

// ── HTTP server ───────────────────────────────────────────────
const server = http.createServer(async (req, res) => {
  setCORSHeaders(res);

  // preflight
  if (req.method === 'OPTIONS') {
    res.writeHead(204);
    res.end();
    return;
  }

  // track client
  activeClients.add(res);
  res.on('close', () => activeClients.delete(res));

  const url = new URL(req.url, `http://localhost:${PORT}`);

  try {
    // ── Health
    if (req.method === 'GET' && url.pathname === '/health') {
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({
        status:        'ok',
        activeClients: activeClients.size,
        defaultModel:  DEFAULT_MODEL,
      }));
      return;
    }

    // ── OpenAI endpoint
    if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
      await handleOpenAI(req, res);
      return;
    }

    // ── Anthropic endpoint
    if (req.method === 'POST' && url.pathname === '/v1/messages') {
      await handleAnthropic(req, res);
      return;
    }

    sendError(res, 404, `Unknown route: ${req.method} ${url.pathname}`);

  } catch (err) {
    console.error('[ERROR]', err.message);
    sendError(res, err.status || 502, err.message);
  }
});

server.listen(PORT, () => {
  console.log(`
╔══════════════════════════════════════════════════════╗
║          NVIDIA Nemotron Proxy  –  Running           ║
╠══════════════════════════════════════════════════════╣
║  OpenAI   →  POST http://localhost:${PORT}/v1/chat/completions
║  Anthropic→  POST http://localhost:${PORT}/v1/messages
║  Health   →  GET  http://localhost:${PORT}/health
╠══════════════════════════════════════════════════════╣
║  Default model : ${DEFAULT_MODEL}
║  API key       : ${NVIDIA_API_KEY.slice(0, 8)}...
╚══════════════════════════════════════════════════════╝
`);
});

// ── Graceful shutdown ─────────────────────────────────────────
process.on('SIGINT',  shutdown);
process.on('SIGTERM', shutdown);

function shutdown() {
  console.log('\n[PROXY] Shutting down gracefully...');
  for (const res of activeClients) {
    try { res.end(); } catch {}
  }
  server.close(() => {
    console.log('[PROXY] Server closed.');
    process.exit(0);
  });
}
