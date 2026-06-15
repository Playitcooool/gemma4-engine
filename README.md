# Gemma 4 E4B Engine

Small CLI-first MLX inference engine for Gemma 4 E4B MLX checkpoints, optimized for:

```text
/Volumes/Samsung/lmstudio/lmstudio-community/unsloth:gemma-4-E4B-it-UD-MLX-4bit
```

The runtime uses `mlx_lm` directly. The supported commands are:

- `gemma4 infer`
- `gemma4 serve`
- `gemma4 bench`

## Setup

```bash
uv venv
uv sync
```

## CLI

Run one prompt:

```bash
gemma4 infer --prompt "Say hi."
gemma4 infer --prompt "Write a haiku about MLX." --max-tokens 128
```

Start a local JSON service:

```bash
gemma4 serve --host 127.0.0.1 --port 8000
```

Call the service:

```bash
curl -s http://127.0.0.1:8000/health
curl -s http://127.0.0.1:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Say hi.","max_tokens":64,"prompt_mode":"chat"}'
```

Run a benchmark:

```bash
gemma4 bench --backend mlx --prompt-tokens 128,512 --decode-tokens 64 --warmups 1 --runs 3
```

Use `--json` when you need the full benchmark payload:

```bash
gemma4 bench --backend mlx --prompt-tokens 128,512 --decode-tokens 64 --warmups 1 --runs 3 --json
```

Interactive local chat loads the model once and keeps a session KV cache alive:

```bash
gemma4 chat --model /path/to/model
```

Inside chat, use `/reset`, `/stats`, and `/exit`.

## Prefix Cache

For repeated tasks with the same long document or system prefix:

```bash
gemma4 infer \
  --cache-prefix-file shared_context.txt \
  --cache-prefix-mode raw \
  --prompt "$(cat shared_context.txt) Question: summarize the key risks." \
  --prompt-mode raw
```

The prompt should include the exact prefix text for token-for-token equivalence. Tokenized prefixes
are cached in memory and persisted under `.gemma4-cache/prefix-tokens` by default. Prefilled KV
cache state stays in process memory because it is made of live MLX tensors. The in-process prefix
KV cache is LRU-managed and can automatically reuse the longest cached prefix that matches a later
prompt.

Tune or disable the disk token cache:

```bash
gemma4 serve --token-cache-dir /tmp/gemma4-prefix-tokens
gemma4 serve --token-cache-max-disk-mb 250
gemma4 infer --token-cache-dir "" --prompt "Say hi."
```

## Sessions

The server can keep an append-only KV cache per local session. This is useful for single-user chat
or agent loops where each request adds a small follow-up prompt:

```bash
gemma4 serve --enable-sessions --max-sessions 4
```

Then send a session id:

```bash
curl -s http://127.0.0.1:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{"session_id":"main","prompt":"Explain KV cache briefly.","max_tokens":64}'

curl -s http://127.0.0.1:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{"session_id":"main","prompt":"Now give one example.","max_tokens":64}'
```

The second request reuses the session KV cache and only prefills the newly appended prompt. Use
`"reset_session": true` to clear a specific session before generating.

## Performance Flags

`--backend auto` and `--backend mlx` both use the MLX runtime.

The default `--prefill-step-size auto` is adaptive: 1024-token chunks for short prompts, then 2048,
4096, and 8192 as prompt length grows. The benchmark command reports median prefill tok/s, decode
tok/s, total tok/s, time to first token, peak memory, speculative acceptance rate, and speedups
against the baseline row.

The `single_user_fast` profile applies the local speed defaults from the optimization plan:
adaptive prefill chunks, retained allocator cache, async prefill sync, `custom` streaming decode,
and four session slots. It is the default for `gemma4 chat`; for one-shot inference or the server,
enable it explicitly:

```bash
gemma4 infer --profile single_user_fast --prompt "Say hi."
gemma4 serve --profile single_user_fast
```

Useful benchmark sweeps:

```bash
gemma4 bench \
  --backend mlx \
  --prompt-tokens 512,2048 \
  --decode-tokens 64 \
  --prefill-step-sizes auto,1024,2048,4096 \
  --prefill-cache-policy clear \
  --decode-variants custom

gemma4 bench \
  --backend mlx \
  --prompt-tokens 512 \
  --decode-tokens 64 \
  --prefill-sync-policies eval,async,none \
  --prefill-cache-policy clear \
  --decode-variants custom

gemma4 bench \
  --backend mlx \
  --prompt-tokens 512 \
  --decode-tokens 128 \
  --decode-variants custom,custom_blockwise_8,custom_blockwise_16,custom_blockwise_32

gemma4 bench \
  --backend mlx \
  --prompt-tokens 512 \
  --decode-tokens 128 \
  --decode-variants custom,custom_speculative_ngram \
  --speculative-ngram-min 3 \
  --speculative-ngram-max 6 \
  --speculative-draft-tokens 4
```

`--prefill-cache-policy retain` keeps the MLX allocator cache between prefill chunks. It can improve
prefill throughput on some runs but may spend more memory, so keep it benchmark-gated.

For lower synchronization overhead, `--prefill-sync-policy periodic --prefill-sync-every 4`
evaluates cache state every few chunks and on the final chunk. For memory-aware cache clearing,
`--prefill-cache-policy threshold --prefill-cache-threshold-gb 18` clears only after active MLX
memory crosses the threshold.

For machines with enough unified memory, the CLI can raise MLX memory residency/cache limits before
loading the model:

```bash
gemma4 serve \
  --mlx-memory-limit-gb 48 \
  --mlx-cache-limit-gb 40 \
  --mlx-wired-limit-gb 32
```
