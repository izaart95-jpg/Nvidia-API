/**
 * NVIDIA Proxy Server
 * ─────────────────────────────────────────────────────────────
 * OpenAI-compatible : POST /v1/chat/completions
 * Anthropic-compat  : POST /v1/messages
 * Health            : GET  /health
 * ─────────────────────────────────────────────────────────────
 * Env vars:
 *   NVIDIA_API_KEY   (required)
 *   PORT             (default 3000)
 *   DEBUG_BUFFER     (default false) - Buffer complete response before streaming
 *   LOG_REQUESTS     (default false) - Log request bodies to files
 *   LOG_RESPONSES    (default false) - Log raw NVIDIA response chunks to files
 *   LOG_DIR          (default "./logs") - Directory for log files
 *
 * Fixes:
 *   1. reasoning_content preserved in thinking blocks (streaming)
 *   2. message_start fires only on first real content chunk (state-based)
 *   3. Leftover buffer flushed after stream ends — fixes truncated output
 *   4. Token counts correct — NVIDIA sends usage in a trailing chunk with
 *      choices:[] which was being skipped; now captured and emitted properly
 *   5. Proper Anthropic SSE format with separate thinking and text blocks
 *   6. Handle <think> tags in content - route to thinking block
 */

import http from 'http';
import { URL } from 'url';
import fs from 'fs';
import path from 'path';
import { randomUUID } from 'crypto';

// ── Config ────────────────────────────────────────────────────
const PORT            = parseInt(process.env.PORT || '3000', 10);
const NVIDIA_API_KEY  = process.env.NVIDIA_API_KEY;
const NVIDIA_BASE     = 'https://integrate.api.nvidia.com/v1';
const DEFAULT_MODEL   = 'nvidia/nemotron-3-super-120b-a12b';
const DEFAULT_MAX_TOK = 16384;
const DEBUG_BUFFER    = process.env.DEBUG_BUFFER === 'true';
const LOG_REQUESTS    = process.env.LOG_REQUESTS === 'true';
const LOG_RESPONSES   = process.env.LOG_RESPONSES === 'true';
const LOG_DIR         = process.env.LOG_DIR || './logs';

if (!NVIDIA_API_KEY) {
  console.error('[ERROR] NVIDIA_API_KEY env var is required');
  process.exit(1);
}

// ── Initialize logging directory ──────────────────────────────
if (LOG_REQUESTS || LOG_RESPONSES) {
  if (!fs.existsSync(LOG_DIR)) {
    fs.mkdirSync(LOG_DIR, { recursive: true });
  }
  console.log(`[LOG] Logging enabled. Directory: ${path.resolve(LOG_DIR)}`);
}

// ── Logger utility ────────────────────────────────────────────
class RequestLogger {
  constructor(requestId, endpoint) {
    this.requestId = requestId;
    this.endpoint = endpoint;
    this.timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    this.reqLogFile = null;
    this.resLogFile = null;
    this.chunkIndex = 0;
    
    if (LOG_REQUESTS) {
      this.reqLogFile = path.join(LOG_DIR, `${this.timestamp}_${this.requestId}_${endpoint}_request.json`);
    }
    
    if (LOG_RESPONSES) {
      this.resLogFile = path.join(LOG_DIR, `${this.timestamp}_${this.requestId}_${endpoint}_response_chunks.jsonl`);
    }
  }
  
  logRequest(body, headers) {
    if (!LOG_REQUESTS || !this.reqLogFile) return;
    
    const logData = {
      requestId: this.requestId,
      timestamp: new Date().toISOString(),
      endpoint: this.endpoint,
      headers: {
        'content-type': headers['content-type'],
        'user-agent': headers['user-agent'],
      },
      body: body,
    };
    
    try {
      fs.writeFileSync(this.reqLogFile, JSON.stringify(logData, null, 2));
      console.log(`[LOG] Request saved: ${this.reqLogFile}`);
    } catch (err) {
      console.error(`[LOG ERROR] Failed to write request log: ${err.message}`);
    }
  }
  
  logResponseChunk(chunkData, isRaw = true) {
    if (!LOG_RESPONSES || !this.resLogFile) return;
    
    const logEntry = {
      requestId: this.requestId,
      chunkIndex: this.chunkIndex++,
      timestamp: new Date().toISOString(),
      type: isRaw ? 'raw_chunk' : 'processed_event',
      data: chunkData,
    };
    
    try {
      fs.appendFileSync(this.resLogFile, JSON.stringify(logEntry) + '\n');
    } catch (err) {
      console.error(`[LOG ERROR] Failed to write response chunk log: ${err.message}`);
    }
  }
  
  logComplete() {
    if (LOG_RESPONSES && this.resLogFile) {
      console.log(`[LOG] Response chunks saved: ${this.resLogFile} (${this.chunkIndex} chunks)`);
    }
  }
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

  if (body.system) {
    messages.push({ role: 'system', content: body.system });
  }

  for (const msg of (body.messages || [])) {
    if (typeof msg.content === 'string') {
      messages.push({ role: msg.role, content: msg.content });
    } else if (Array.isArray(msg.content)) {
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
    stream:               opts.stream !== false,
  };
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

// ── Helper: Parse content for <think> tags ───────────────────
function parseContentForThinking(content) {
  // Pattern to match <think>...</think> tags
  const thinkPattern = /<think>([\s\S]*?)<\/think>/g;
  const thinks = [];
  let textContent = content;
  let match;
  
  while ((match = thinkPattern.exec(content)) !== null) {
    thinks.push(match[1]);
    textContent = textContent.replace(match[0], '');
  }
  
  // Also handle incomplete tags (like <think> without closing)
  const incompleteThinkMatch = content.match(/<think>([\s\S]*?)$/);
  if (incompleteThinkMatch && !content.includes('</think>')) {
    thinks.push(incompleteThinkMatch[1]);
    textContent = textContent.replace(/<think>[\s\S]*?$/, '');
  }
  
  // Handle closing tag without opening (rare, but possible)
  const closingThinkMatch = content.match(/^([\s\S]*?)<\/think>/);
  if (closingThinkMatch && !content.includes('<think>')) {
    thinks.push(closingThinkMatch[1]);
    textContent = textContent.replace(/^[\s\S]*?<\/think>/, '');
  }
  
  return {
    thinking: thinks.join(''),
    text: textContent.trim()
  };
}

// ── Core: stream upstream → client (OpenAI passthrough) ──────
async function streamOpenAI(res, payload, logger = null) {
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

      const chunk = decoder.decode(value, { stream: true });
      
      // Log raw chunk
      if (logger) {
        logger.logResponseChunk(chunk);
      }
      
      buffer += chunk;
      const lines = buffer.split('\n');
      buffer = lines.pop();

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
    if (buffer.trim()) res.write(`${buffer}\n`);
  } finally {
    reader.releaseLock();
    if (logger) logger.logComplete();
    res.end();
  }
}

// ── Core: stream upstream → client (Anthropic SSE) ───────────
async function streamAnthropic(res, payload, logger = null) {
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

  // Per-request state
  const state = {
    messageStarted:      false,
    blockIndex:          0,
    currentBlockType:    null,   // 'thinking' | 'text' | null
    messageId:           null,
    model:               null,
    pendingFinish:       null,   // { stop_reason: string }
    inputTokens:         0,
    outputTokens:        0,
    // Buffer for incomplete think tags
    thinkBuffer:         '',
    inThinkTag:          false,
  };

  function writeEvent(eventType, data) {
    const eventString = `event: ${eventType}\ndata: ${JSON.stringify(data)}\n\n`;
    res.write(eventString);
    
    // Log processed event
    if (logger) {
      logger.logResponseChunk({ eventType, data }, false);
    }
  }

  // Emit the queued finish events with real token counts
  function flushPendingFinish(inputTokens, outputTokens) {
    if (!state.pendingFinish) return;
    writeEvent('message_delta', {
      type:  'message_delta',
      delta: { stop_reason: state.pendingFinish.stop_reason },
      usage: {
        input_tokens:  inputTokens,
        output_tokens: outputTokens,
      },
    });
    writeEvent('message_stop', { type: 'message_stop' });
    state.pendingFinish = null;
  }

  function emitThinkingDelta(thinkingText) {
    if (!thinkingText) return;
    
    // Start thinking block if not already in one
    if (state.currentBlockType !== 'thinking') {
      // Close text block if open
      if (state.currentBlockType === 'text') {
        writeEvent('content_block_stop', {
          type: 'content_block_stop',
          index: state.blockIndex,
        });
        state.blockIndex++;
        state.currentBlockType = null;
      }
      
      writeEvent('content_block_start', {
        type:  'content_block_start',
        index: state.blockIndex,
        content_block: { type: 'thinking', thinking: '' },
      });
      state.currentBlockType = 'thinking';
    }

    // Emit thinking delta
    writeEvent('content_block_delta', {
      type:  'content_block_delta',
      index: state.blockIndex,
      delta: { type: 'thinking_delta', thinking: thinkingText },
    });
  }

  function emitTextDelta(textContent) {
    if (!textContent) return;
    
    // Close thinking block if open
    if (state.currentBlockType === 'thinking') {
      writeEvent('content_block_stop', {
        type: 'content_block_stop',
        index: state.blockIndex,
      });
      state.blockIndex++;
      state.currentBlockType = null;
    }

    // Start text block if not already in one
    if (state.currentBlockType !== 'text') {
      writeEvent('content_block_start', {
        type:  'content_block_start',
        index: state.blockIndex,
        content_block: { type: 'text', text: '' },
      });
      state.currentBlockType = 'text';
    }

    // Emit text delta
    writeEvent('content_block_delta', {
      type:  'content_block_delta',
      index: state.blockIndex,
      delta: { type: 'text_delta', text: textContent },
    });
  }

  function processContent(content) {
    // Combine with existing buffer
    const fullContent = state.thinkBuffer + content;
    
    // Look for think tags
    const thinkStart = fullContent.indexOf('<think>');
    const thinkEnd = fullContent.indexOf('</think>');
    
    if (thinkStart === -1 && thinkEnd === -1) {
      // No think tags at all
      if (!state.inThinkTag) {
        emitTextDelta(fullContent);
        state.thinkBuffer = '';
      } else {
        // We're inside a think tag, emit as thinking
        emitThinkingDelta(fullContent);
        state.thinkBuffer = '';
      }
      return;
    }
    
    if (thinkStart !== -1 && !state.inThinkTag) {
      // Found start of think tag
      const beforeThink = fullContent.substring(0, thinkStart);
      const afterThinkStart = fullContent.substring(thinkStart + 7); // 7 = '<think>'.length
      
      if (beforeThink) {
        emitTextDelta(beforeThink);
      }
      
      state.inThinkTag = true;
      
      if (thinkEnd !== -1) {
        // Complete think tag
        const thinkContent = afterThinkStart.substring(0, afterThinkStart.indexOf('</think>'));
        const afterThinkEnd = afterThinkStart.substring(afterThinkStart.indexOf('</think>') + 8); // 8 = '</think>'.length
        
        if (thinkContent) {
          emitThinkingDelta(thinkContent);
        }
        
        state.inThinkTag = false;
        state.thinkBuffer = afterThinkEnd;
        
        // Process remaining content recursively
        if (afterThinkEnd) {
          processContent('');
        }
      } else {
        // Incomplete think tag, buffer for next chunk
        state.thinkBuffer = afterThinkStart;
      }
    } else if (thinkEnd !== -1 && state.inThinkTag) {
      // Found end of think tag
      const thinkContent = fullContent.substring(0, thinkEnd);
      const afterThinkEnd = fullContent.substring(thinkEnd + 8); // 8 = '</think>'.length
      
      if (thinkContent) {
        emitThinkingDelta(thinkContent);
      }
      
      state.inThinkTag = false;
      state.thinkBuffer = afterThinkEnd;
      
      // Process remaining content
      if (afterThinkEnd) {
        processContent('');
      }
    } else if (state.inThinkTag) {
      // Still inside think tag, emit everything as thinking
      emitThinkingDelta(fullContent);
      state.thinkBuffer = '';
    } else {
      // No active think tag, emit as text
      emitTextDelta(fullContent);
      state.thinkBuffer = '';
    }
  }

  function processChunk(chunk) {
    // Handle usage chunk (empty choices)
    if (Array.isArray(chunk.choices) && chunk.choices.length === 0) {
      if (chunk.usage) {
        state.inputTokens  = chunk.usage.prompt_tokens     ?? 0;
        state.outputTokens = chunk.usage.completion_tokens ?? 0;
        flushPendingFinish(state.inputTokens, state.outputTokens);
      }
      return;
    }

    const choice = chunk.choices?.[0];
    if (!choice) return;

    const delta = choice.delta || {};

    // Capture id/model early for message_start
    if (!state.messageId && chunk.id)    state.messageId = chunk.id;
    if (!state.model    && chunk.model)  state.model     = chunk.model;

    // Handle finish reason
    if (choice.finish_reason) {
      if (state.currentBlockType) {
        writeEvent('content_block_stop', {
          type: 'content_block_stop',
          index: state.blockIndex,
        });
        state.currentBlockType = null;
      }
      state.pendingFinish = {
        stop_reason: choice.finish_reason === 'stop' ? 'end_turn' : choice.finish_reason,
      };
      return;
    }

    // Extract reasoning and content
    const reasoning = delta.reasoning_content || delta.reasoning || '';
    const content   = delta.content || '';

    // Start message if needed
    if (!state.messageStarted && (reasoning || content)) {
      writeEvent('message_start', {
        type: 'message_start',
        message: {
          id:          state.messageId,
          type:        'message',
          role:        'assistant',
          model:       state.model,
          content:     [],
          stop_reason: null,
          usage:       { input_tokens: 0, output_tokens: 0 },
        },
      });
      writeEvent('ping', { type: 'ping' });
      state.messageStarted = true;
      state.blockIndex = 0;
    }

    // Handle reasoning content (explicit field)
    if (reasoning) {
      emitThinkingDelta(reasoning);
    }

    // Handle content (may contain <think> tags)
    if (content) {
      processContent(content);
    }
  }

  function processLine(line) {
    const trimmed = line.trim();
    if (!trimmed || trimmed === 'data: [DONE]') return;
    if (!trimmed.startsWith('data: ')) return;
    try {
      const parsedChunk = JSON.parse(trimmed.slice(6));
      
      // Log parsed chunk
      if (logger) {
        logger.logResponseChunk(parsedChunk, true);
      }
      
      processChunk(parsedChunk);
    } catch (e) {
      console.error('[ERROR] Failed to parse chunk:', e.message);
    }
  }

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const chunk = decoder.decode(value, { stream: true });
      buffer += chunk;
      const lines = buffer.split('\n');
      buffer = lines.pop();

      for (const line of lines) processLine(line);
    }

    // Flush leftover buffer
    if (buffer.trim()) processLine(buffer);
    
    // Flush any remaining think buffer
    if (state.thinkBuffer) {
      if (state.inThinkTag) {
        emitThinkingDelta(state.thinkBuffer);
      } else {
        emitTextDelta(state.thinkBuffer);
      }
    }

    // Close any open block
    if (state.currentBlockType) {
      writeEvent('content_block_stop', {
        type: 'content_block_stop',
        index: state.blockIndex,
      });
    }

    // Emit finish events
    flushPendingFinish(state.inputTokens, state.outputTokens);

  } finally {
    reader.releaseLock();
    if (logger) logger.logComplete();
    res.end();
  }
}

// ── DEBUG: Buffer complete response then stream as SSE ───────
async function debugBufferAnthropic(res, payload, logger = null) {
  console.log('[DEBUG] Buffering complete response before streaming...');
  const startTime = Date.now();
  
  // Force non-streaming from upstream
  const nonStreamPayload = { ...payload, stream: false };
  const upstreamRes = await fetchUpstream(nonStreamPayload);
  const data = await upstreamRes.json();
  
  const endTime = Date.now();
  console.log(`[DEBUG] Response buffered in ${endTime - startTime}ms`);

  res.writeHead(200, {
    'Content-Type':      'text/event-stream',
    'Cache-Control':     'no-cache',
    'Connection':        'keep-alive',
    'X-Accel-Buffering': 'no',
  });

  const choice = data.choices?.[0] || {};
  const message = choice.message || {};
  let content = message.content || '';
  let reasoning = message.reasoning_content || message.reasoning || '';
  
  // Parse <think> tags from content
  const parsed = parseContentForThinking(content);
  if (parsed.thinking) {
    reasoning = reasoning + parsed.thinking;
    content = parsed.text;
  }
  
  const messageId = data.id || `msg_${Date.now()}`;
  const model = data.model || payload.model;
  const finishReason = choice.finish_reason === 'stop' ? 'end_turn' : choice.finish_reason;
  const inputTokens = data.usage?.prompt_tokens ?? 0;
  const outputTokens = data.usage?.completion_tokens ?? 0;

  // Helper to write SSE events
  function writeEvent(eventType, eventData) {
    const eventString = `event: ${eventType}\ndata: ${JSON.stringify(eventData)}\n\n`;
    res.write(eventString);
    
    // Log processed event
    if (logger) {
      logger.logResponseChunk({ eventType, data: eventData }, false);
    }
  }

  // Emit message_start
  writeEvent('message_start', {
    type: 'message_start',
    message: {
      id: messageId,
      type: 'message',
      role: 'assistant',
      model: model,
      content: [],
      stop_reason: null,
      usage: { input_tokens: 0, output_tokens: 0 },
    },
  });
  
  writeEvent('ping', { type: 'ping' });

  let blockIndex = 0;

  // Emit thinking block if reasoning exists
  if (reasoning) {
    writeEvent('content_block_start', {
      type: 'content_block_start',
      index: blockIndex,
      content_block: { type: 'thinking', thinking: '' },
    });

    // Split reasoning into chunks for realistic streaming simulation
    const chunkSize = 10;
    for (let i = 0; i < reasoning.length; i += chunkSize) {
      const chunk = reasoning.slice(i, i + chunkSize);
      writeEvent('content_block_delta', {
        type: 'content_block_delta',
        index: blockIndex,
        delta: { type: 'thinking_delta', thinking: chunk },
      });
      // Small delay to simulate streaming
      await new Promise(resolve => setTimeout(resolve, 5));
    }

    writeEvent('content_block_stop', {
      type: 'content_block_stop',
      index: blockIndex,
    });
    blockIndex++;
  }

  // Emit text block if content exists
  if (content) {
    writeEvent('content_block_start', {
      type: 'content_block_start',
      index: blockIndex,
      content_block: { type: 'text', text: '' },
    });

    // Split content into words for realistic streaming simulation
    const words = content.split(/(\s+)/);
    for (const word of words) {
      if (word) {
        writeEvent('content_block_delta', {
          type: 'content_block_delta',
          index: blockIndex,
          delta: { type: 'text_delta', text: word },
        });
        // Small delay to simulate streaming
        await new Promise(resolve => setTimeout(resolve, 10));
      }
    }

    writeEvent('content_block_stop', {
      type: 'content_block_stop',
      index: blockIndex,
    });
  }

  // Emit final events
  writeEvent('message_delta', {
    type: 'message_delta',
    delta: { stop_reason: finishReason },
    usage: {
      input_tokens: inputTokens,
      output_tokens: outputTokens,
    },
  });
  
  writeEvent('message_stop', { type: 'message_stop' });
  
  res.end();
  if (logger) logger.logComplete();
  console.log('[DEBUG] Stream simulation complete');
}

// ── Core: non-streaming (OpenAI passthrough) ─────────────────
async function fullOpenAI(res, payload, logger = null) {
  const upstreamRes = await fetchUpstream(payload);
  const data        = await upstreamRes.json();
  
  // Log response
  if (logger) {
    logger.logResponseChunk(data, true);
    logger.logComplete();
  }
  
  res.writeHead(200, { 'Content-Type': 'application/json' });
  res.end(JSON.stringify(data));
}

// ── Core: non-streaming (Anthropic) ──────────────────────────
async function fullAnthropic(res, payload, logger = null) {
  const upstreamRes = await fetchUpstream(payload);
  const data        = await upstreamRes.json();
  const choice      = data.choices?.[0] || {};
  const message     = choice.message || {};
  let content       = message.content || '';
  let reasoning     = message.reasoning_content || message.reasoning || '';
  
  // Parse <think> tags from content
  const parsed = parseContentForThinking(content);
  if (parsed.thinking) {
    reasoning = reasoning + parsed.thinking;
    content = parsed.text;
  }

  // Log response
  if (logger) {
    logger.logResponseChunk(data, true);
  }

  const response = {
    id:          data.id,
    type:        'message',
    role:        'assistant',
    model:       data.model,
    content:     [{ type: 'text', text: content }],
    stop_reason: choice.finish_reason === 'stop' ? 'end_turn' : choice.finish_reason,
    usage: {
      input_tokens:  data.usage?.prompt_tokens     ?? 0,
      output_tokens: data.usage?.completion_tokens ?? 0,
    },
  };
  
  // Add thinking if present (for clients that support it)
  if (reasoning) {
    response.thinking = reasoning;
  }
  
  // Log transformed response
  if (logger) {
    logger.logResponseChunk({ transformed: response }, false);
    logger.logComplete();
  }
  
  res.writeHead(200, { 'Content-Type': 'application/json' });
  res.end(JSON.stringify(response));
}

// ── Route: POST /v1/chat/completions  (OpenAI-compatible) ────
async function handleOpenAI(req, res) {
  const body   = await readBody(req);
  const stream = body.stream === true;
  const payload = buildUpstreamPayload(body.messages || [], {
    ...body,
    model: body.model || DEFAULT_MODEL,
    stream,
  });

  const requestId = randomUUID().slice(0, 8);
  const logger = new RequestLogger(requestId, 'openai');
  
  // Log request
  logger.logRequest(body, req.headers);

  console.log(`[OpenAI]    id=${requestId} model=${payload.model} stream=${stream} client=${req.socket.remoteAddress}`);

  if (stream) {
    await streamOpenAI(res, payload, logger);
  } else {
    await fullOpenAI(res, payload, logger);
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

  const requestId = randomUUID().slice(0, 8);
  const logger = new RequestLogger(requestId, 'anthropic');
  
  // Log request
  logger.logRequest(body, req.headers);

  console.log(`[Anthropic] id=${requestId} model=${payload.model} stream=${stream} debug=${DEBUG_BUFFER} client=${req.socket.remoteAddress}`);

  if (stream) {
    if (DEBUG_BUFFER) {
      await debugBufferAnthropic(res, payload, logger);
    } else {
      await streamAnthropic(res, payload, logger);
    }
  } else {
    await fullAnthropic(res, payload, logger);
  }
}

// ── HTTP server ───────────────────────────────────────────────
const server = http.createServer(async (req, res) => {
  setCORSHeaders(res);

  if (req.method === 'OPTIONS') {
    res.writeHead(204);
    res.end();
    return;
  }

  activeClients.add(res);
  res.on('close', () => activeClients.delete(res));

  const url = new URL(req.url, `http://localhost:${PORT}`);

  try {
    if (req.method === 'GET' && url.pathname === '/health') {
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({
        status:        'ok',
        activeClients: activeClients.size,
        defaultModel:  DEFAULT_MODEL,
        debugBuffer:   DEBUG_BUFFER,
        logging: {
          requests: LOG_REQUESTS,
          responses: LOG_RESPONSES,
          directory: path.resolve(LOG_DIR),
        },
      }));
      return;
    }

    if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
      await handleOpenAI(req, res);
      return;
    }

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
║          NVIDIA Proxy  –  Running           ║
╠══════════════════════════════════════════════════════╣
║  OpenAI   →  POST http://localhost:${PORT}/v1/chat/completions
║  Anthropic→  POST http://localhost:${PORT}/v1/messages
║  Health   →  GET  http://localhost:${PORT}/health
╠══════════════════════════════════════════════════════╣
║  Default model : ${DEFAULT_MODEL}
║  API key       : ${NVIDIA_API_KEY.slice(0, 8)}...
║  Debug buffer  : ${DEBUG_BUFFER ? 'ENABLED' : 'disabled'}
╠══════════════════════════════════════════════════════╣
║  Logging:
║    Requests    : ${LOG_REQUESTS ? 'ENABLED' : 'disabled'}
║    Responses   : ${LOG_RESPONSES ? 'ENABLED' : 'disabled'}
║    Log dir     : ${path.resolve(LOG_DIR)}
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
