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
uv sync --extra dev
```

## CLI

```bash
gemma4 infer --prompt "Write a haiku about MLX."
gemma4 bench
gemma4 compare --baseline mlx_lm --prompt "Explain KV cache in one sentence."
gemma4 serve --host 127.0.0.1 --port 8000
```

Useful flags:

```bash
gemma4 infer --backend auto --max-tokens 128 --prefill-step-size auto
gemma4 bench --prompt-tokens 128,512,2048,8192 --decode-tokens 128,512 --json
gemma4 compare --backend auto --prompt "Say hi." --max-tokens 64
```

`--backend auto` uses Rust kernels only after a self-test passes. If the extension is unavailable,
incorrect, or slower for the local operation, it falls back to MLX.

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

The default `--prefill-step-size auto` uses smaller chunks for long prompts to reduce memory
pressure. On the local target model, the verified fast path is:

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
