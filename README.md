# Gemma 4 E4B Engine

Small CLI-first MLX inference engine for Gemma 4 E4B MLX checkpoints, optimized first for:

```text
/Volumes/Samsung/lmstudio/lmstudio-community/unsloth:gemma-4-E4B-it-UD-MLX-4bit
```

Python/MLX handles model loading and execution through `mlx_lm`. Optional Rust/Metal kernels live in
`crates/gemma4-kernels` and are selected only when available and validated.

## Setup

```bash
uv venv
uv sync
```

## CLI

The default flow is intentionally small:

```bash
gemma4 serve
gemma4 infer --prompt "Say hi."
```

Useful flags:

```bash
gemma4 serve --host 127.0.0.1 --port 8000
gemma4 infer --prompt "Write a haiku about MLX." --max-tokens 128
gemma4 infer --backend auto --max-tokens 128 --prefill-step-size auto
gemma4 bench --prompt-tokens 128,512,2048,8192 --decode-tokens 128,512 --json
gemma4 compare --backend auto --prompt "Say hi." --max-tokens 64
```

`--backend auto` keeps the production generation path on MLX. Rust argmax is currently a
correctness/testing path and is used only when explicitly requested or after it can beat the MLX
path without CPU copies. `--backend auto` never selects speculative decoding.

For development tools:

```bash
uv sync --extra dev
```

## Performance Notes

For deployable throughput, keep the model loaded and use the `bench` command or `Gemma4Engine`
instead of repeatedly starting a new process. The benchmark path reuses one loaded model across
warmups and measured runs.

For a persistent local service, run:

```bash
gemma4 serve --backend auto --host 127.0.0.1 --port 8000
```

Then call:

```bash
curl -s http://127.0.0.1:8000/health
curl -s http://127.0.0.1:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Say hi.","max_tokens":64,"prompt_mode":"chat"}'
```

The service loads the model once at startup and serializes generation with a process-local lock.
This keeps throughput stable for repeated task requests while avoiding concurrent mutation of the
MLX KV cache state.

For repeated tasks with the same long document or system prefix, enable prefix caching:

```bash
gemma4 infer \
  --cache-prefix-file shared_context.txt \
  --cache-prefix-mode raw \
  --prompt "$(cat shared_context.txt) Question: summarize the key risks." \
  --prompt-mode raw
```

The prompt should include the exact prefix text for token-for-token equivalence. Tokenizers can
merge text across prefix/suffix boundaries, so independently tokenizing a prefix and suffix is not
always identical to tokenizing the concatenated prompt. On repeated exact-prefix requests, the
engine reuses the prefetched KV cache and only prefills the suffix.

Prefix caching is hierarchical:

- Tokenized prefixes are cached in process memory first.
- Tokenized prefixes are also persisted on disk under `.gemma4-cache/prefix-tokens` by default, so
  short-lived `gemma4 infer` processes can avoid retokenizing repeated prefixes.
- Prefilled KV cache state stays in process memory because it is made of live MLX cache tensors; this
  is the fastest and least brittle representation for serving.

To change or disable the disk token cache:

```bash
gemma4 serve --token-cache-dir /tmp/gemma4-prefix-tokens
gemma4 infer --token-cache-dir "" --prompt "Say hi."
```

The default `--prefill-step-size auto` uses 512-token prefill chunks. On the local target model,
512 beat larger chunk candidates in a bounded prefill-focused benchmark at 2048 and 8192 prompt
tokens while also using less peak memory. The verified smoke path is:

```bash
gemma4 bench --backend mlx --prompt-tokens 128,512,2048,8192 --decode-tokens 64 --warmups 1 --runs 3
```

For explicit long-context tuning:

```bash
gemma4 bench --backend mlx --prompt-tokens 8192 --decode-tokens 64 --prefill-step-size 1024 --json
```

`--backend rust-metal` is currently a correctness/testing path for the Rust extension. The
production fast path is `--backend auto` or `--backend mlx` until the Rust argmax kernel is fully
GPU-resident and wins the median-speed gate.

`--kv-bits` is exposed for compatible MLX models, but it is automatically disabled for the current
Gemma 4 shared-KV checkpoint because upstream `mlx_lm` rotating/shared cache quantization is not
compatible with this architecture yet.

## Speculative Decoding

Speculative decoding is experimental and not part of the default install or serving path. Install
the optional dependencies only when testing a drafter:

```bash
uv sync --extra speculative
```

For package installs, use:

```bash
pip install 'gemma4-engine[speculative]'
```

Gemma 4 assistant/MTP drafters can be tested explicitly:

```bash
gemma4 infer \
  --backend mlx \
  --draft-model /Volumes/Samsung/lmstudio/lmstudio-community/mlx-community:gemma-4-E4B-it-qat-assistant-4bit \
  --draft-tokens 4 \
  --prompt "Say hi."
```

This path uses the optional `mlx-vlm` package for Gemma 4 MTP hooks and includes a local
compatibility patch for the QAT assistant's quantized sparse embedding head. It is intentionally
opt-in: on the local
`gemma-4-E4B-it-qat-assistant-4bit` drafter, measured decode was slower than the default MLX path
even with high acceptance, because the current integration loads an `mlx-vlm` target model in
addition to the normal `mlx-lm` target and pays extra MTP runtime overhead.

Use `--draft-model` for experiments and acceptance-rate reporting, not as the production default
unless local benchmarks show it wins:

```bash
gemma4 bench \
  --backend mlx \
  --draft-model /Volumes/Samsung/lmstudio/lmstudio-community/mlx-community:gemma-4-E4B-it-qat-assistant-4bit \
  --draft-tokens 4 \
  --prompt-tokens 128,512,2048 \
  --decode-tokens 64 \
  --json
```
