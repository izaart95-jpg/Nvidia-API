# NVIDIA Proxy

A lightweight Node.js proxy server that exposes **NVIDIA's** models behind two drop-in compatible API endpoints — one for OpenAI clients and one for Anthropic clients — with true live SSE streaming, parallel client support, and graceful shutdown.

---

## Files

```
nvidia.js      ← main proxy server
package.json   ← package file
README.md      ← this file
```

---

## Requirements

- Node.js 18+ (native `fetch` + `ReadableStream`)
- An [NVIDIA API key](https://integrate.api.nvidia.com)
- No npm dependencies — uses only Node built-ins

---

## Setup

```bash
export NVIDIA_API_KEY=your_key_here
node nvidia.js
```

Custom port:

```bash
PORT=8080 NVIDIA_API_KEY=your_key node nvidia.js
```

---

## Endpoints

| Method | Path | Compatible with |
|--------|------|----------------|
| `POST` | `/v1/chat/completions` | OpenAI SDK / any OAI client |
| `POST` | `/v1/messages` | Anthropic SDK |
| `GET`  | `/health` | Status + active client count |

---

## Request Options

All fields are optional unless noted.

### Common fields (both endpoints)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | string | `nvidia/nemotron-3-super-120b-a12b` | Model to use |
| `stream` | boolean | `true` | Stream SSE chunks or return full JSON |
| `temperature` | number | `1` | Sampling temperature |
| `top_p` | number | `0.95` | Nucleus sampling |
| `max_tokens` | number | `16384` | Max output tokens |

### OpenAI endpoint extras

| Field | Type | Description |
|-------|------|-------------|
| `messages` | array | **Required.** `[{role, content}]` |

### Anthropic endpoint extras

| Field | Type | Description |
|-------|------|-------------|
| `messages` | array | **Required.** `[{role, content}]` — content can be a string or block array |
| `system` | string | System prompt |

---

## curl Examples

### Health check

```bash
curl http://localhost:3000/health
```

---

### OpenAI — streaming (default)

```bash
curl -N http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Count 1 to 10"}]}'
```

### OpenAI — non-streaming

```bash
curl http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Say hi"}], "stream": false}'
```

### OpenAI — custom model + system prompt

```bash
curl -N http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nvidia/nemotron-3-super-120b-a12b",
    "messages": [
      {"role": "system", "content": "You are a pirate."},
      {"role": "user", "content": "Describe the ocean"}
    ]
  }'
```

---

### Anthropic — streaming (default)

```bash
curl -N http://localhost:3000/v1/messages \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Count 1 to 10"}]}'
```

### Anthropic — non-streaming with system prompt

```bash
curl http://localhost:3000/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "system": "Be concise.",
    "messages": [{"role": "user", "content": "Capital of France?"}],
    "stream": false
  }'
```

### Anthropic — content block format

```bash
curl -N http://localhost:3000/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{
      "role": "user",
      "content": [{"type": "text", "text": "Explain black holes simply"}]
    }]
  }'
```

---

### Parallel clients

Run these simultaneously in separate terminals to test concurrent streaming:

```bash
# Terminal 1
curl -N http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Write a poem about rain"}]}'

# Terminal 2
curl -N http://localhost:3000/v1/messages \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Write a poem about fire"}]}'

# Check active count mid-stream
curl http://localhost:3000/health
# → {"status":"ok","activeClients":2,"defaultModel":"nvidia/nemotron-..."}
```

> **Tip:** The `-N` flag disables curl's own output buffering so you see tokens as they arrive.

---

## Using with SDKs

### OpenAI SDK (Node)

```js
import OpenAI from 'openai';

const client = new OpenAI({
  apiKey: 'any-value',           // proxy ignores auth
  baseURL: 'http://localhost:3000/v1',
});

const stream = await client.chat.completions.create({
  model: 'nvidia/nemotron-3-super-120b-a12b',
  messages: [{ role: 'user', content: 'Hello!' }],
  stream: true,
});

for await (const chunk of stream) {
  process.stdout.write(chunk.choices[0]?.delta?.content ?? '');
}
```

### Anthropic SDK (Node)

```js
import Anthropic from '@anthropic-ai/sdk';

const client = new Anthropic({
  apiKey: 'any-value',           // proxy ignores auth
  baseURL: 'http://localhost:3000',
});

const stream = await client.messages.create({
  model: 'nvidia/nemotron-3-super-120b-a12b',
  max_tokens: 1024,
  messages: [{ role: 'user', content: 'Hello!' }],
  stream: true,
});

for await (const event of stream) {
  if (event.type === 'content_block_delta') {
    process.stdout.write(event.delta.text);
  }
}
```

---

## How Streaming Works

Raw `fetch` with `ReadableStream` is used instead of any SDK, so bytes flow directly from NVIDIA to your client with zero buffering. Each SSE line is parsed individually and re-emitted as a proper `data: {...}\n\n` frame the instant it arrives.

The Anthropic endpoint additionally translates OpenAI chunk format into Anthropic event types in real time:

```
message_start → content_block_start → ping
  → content_block_delta (per token)
→ content_block_stop → message_delta → message_stop
```

---

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `NVIDIA_API_KEY` | ✅ Yes | — | Your NVIDIA API key |
| `PORT` | No | `3000` | Port to listen on |

---

## Graceful Shutdown

`SIGINT` (Ctrl+C) and `SIGTERM` both drain all active SSE streams before closing the server — no connections are dropped mid-response.
