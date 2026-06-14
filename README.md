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
```

Useful flags:

```bash
gemma4 infer --backend auto --max-tokens 128 --prefill-step-size auto
gemma4 bench --prompt-tokens 128,512,2048,8192 --decode-tokens 128,512 --json
```

`--backend auto` uses Rust kernels only after a self-test passes. If the extension is unavailable,
incorrect, or slower for the local operation, it falls back to MLX.
