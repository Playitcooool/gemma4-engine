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
cache state stays in process memory because it is made of live MLX tensors.

Tune or disable the disk token cache:

```bash
gemma4 serve --token-cache-dir /tmp/gemma4-prefix-tokens
gemma4 serve --token-cache-max-disk-mb 250
gemma4 infer --token-cache-dir "" --prompt "Say hi."
```

## Performance Flags

`--backend auto` and `--backend mlx` both use the MLX runtime.

The default `--prefill-step-size auto` uses 512-token prefill chunks. The benchmark command reports
median prefill tok/s, decode tok/s, total tok/s, time to first token, peak memory, and speedups
against the baseline row.

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
```

`--prefill-cache-policy retain` keeps the MLX allocator cache between prefill chunks. It can improve
prefill throughput on some runs but may spend more memory, so keep it benchmark-gated.

For machines with enough unified memory, the CLI can raise MLX memory residency/cache limits before
loading the model:

```bash
gemma4 serve \
  --mlx-memory-limit-gb 48 \
  --mlx-cache-limit-gb 40 \
  --mlx-wired-limit-gb 32
```
