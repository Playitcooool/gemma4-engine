# Gemma4 Single-User Prefill & Decoding Optimization Plan

## 0. Goal

Optimize the current Gemma4 MLX inference engine for **single-user local usage**, especially:

- Faster **prefill** for long prompts, RAG prompts, system prompts, and agent instructions.
- Lower **time to first token (TTFT)**.
- Faster **single-request decoding** without focusing on multi-user throughput.
- Better reuse of prompt/KV cache across repeated and multi-turn requests.

This plan intentionally deprioritizes:

- Multi-user request scheduling.
- Continuous batching.
- Server-side concurrency.
- Complex multi-tenant KV slot management.

For the current use case, the biggest wins should come from:

1. Avoiding repeated model loading.
2. Improving prefill chunk size selection.
3. Reducing unnecessary MLX synchronization and cache clearing.
4. Adding session-level KV cache.
5. Reducing Python/host synchronization in non-streaming decode.
6. Improving benchmark visibility.

---

## 1. Current Engine Observations

Based on the uploaded code, the current engine already has a useful foundation:

- Persistent `Gemma4Engine` object.
- Manual `cache_prefix` support.
- Prefix token cache through `HierarchicalTokenCache`.
- Prefix KV cache through `_get_or_create_prefix_cache`.
- Multiple decode variants:
  - `custom`
  - `custom_no_async`
  - `custom_eval_next`
  - `custom_defer_ids`
  - `mlx_lm_generate_step`
- Benchmark framework for prefill/decode variants.
- CLI and HTTP server interfaces.

However, several current choices are conservative and leave significant single-user latency performance on the table:

### 1.1 `prefill_step_size="auto"` is fixed to 512

Current behavior:

```python
def _prefill_step_size(value: PrefillStepSize, prompt_tokens: int) -> int:
    if value == "auto":
        return 512
    return int(value)
```

This is likely too small for many long-prompt cases. For 4k, 8k, 16k, or larger prompts, using 512-token chunks causes many model calls and many synchronization/cache-management points.

### 1.2 Prefill synchronizes after every chunk

The prefill loop currently calls `_sync_prompt_cache(prompt_cache, prefill_sync_policy)` after every chunk.

The default policy is usually `eval`, which forces host synchronization after each chunk. This can heavily hurt long-prompt prefill throughput.

### 1.3 Prefill may clear MLX cache after every chunk

The prefill loop also calls `_clear_mlx_cache(prefill_cache_policy)` after every chunk.

The default policy is often `clear`. This is safer for memory but slower for local single-user usage when enough memory is available.

### 1.4 Decode still has per-token Python synchronization

The custom decode path does one-token lookahead with `mx.async_eval(next_token)`, but each generated token still needs:

- `mx.eval(token)`
- `token.item()`
- Python-side EOS check
- Python list append

This is acceptable for streaming, but non-streaming generation can be faster with blockwise synchronization.

### 1.5 Existing `custom_defer_ids` is too extreme

`custom_defer_ids` defers token ID extraction until all tokens are generated. This reduces host sync but has drawbacks:

- EOS is checked late.
- It may over-generate after EOS.
- TTFT becomes misleading.
- Very long generations may create large lazy graphs.

A blockwise version is more practical.

### 1.6 Prefix KV cache exists but is manual

The engine supports `cache_prefix`, but the user must explicitly provide it. For local chat/agent usage, a more automatic session-based KV cache is much more useful.

### 1.7 CLI `infer` path reloads model every time

If the user repeatedly runs `gemma4 infer ...`, the model is reloaded for every command. For local usage, the persistent server path should be preferred, or an interactive REPL mode should be added.

---

## 2. Optimization Strategy for Single-User Usage

The target profile should be:

```text
Single-user, low-latency Gemma4 local inference
```

Main goals:

1. Keep model loaded.
2. Avoid re-prefilling repeated context.
3. Use larger prefill chunks when memory allows.
4. Avoid unnecessary per-chunk synchronization.
5. Avoid unnecessary per-chunk cache clearing.
6. Improve non-streaming decode with blockwise synchronization.
7. Add detailed timing metrics to identify the real bottleneck.

---

## 3. Phase 0 — Benchmark and Measurement Improvements

Before changing too much behavior, improve measurement granularity.

### 3.1 Add detailed timing fields to `RunStats`

Add optional fields:

```python
encode_seconds: float | None = None
prefix_token_cache_seconds: float | None = None
prefix_kv_cache_lookup_seconds: float | None = None
prefix_kv_cache_build_seconds: float | None = None
prefix_kv_cache_clone_seconds: float | None = None
prefill_model_seconds: float | None = None
prefill_sync_seconds: float | None = None
prefill_clear_cache_seconds: float | None = None
first_token_eval_seconds: float | None = None
decode_model_seconds: float | None = None
decode_sync_seconds: float | None = None
decode_token_item_seconds: float | None = None
```

Purpose:

- Separate tokenization overhead from model prefill.
- Separate prefix cache clone cost from actual prefill.
- Measure whether `mx.eval(states)` dominates prefill.
- Measure whether `mx.clear_cache()` dominates prefill.
- Measure decode host-sync overhead.

### 3.2 Track per-token decode latency

For local interactive usage, average decode tok/s is not enough.

Add:

```python
decode_token_latency_p50_seconds
decode_token_latency_p95_seconds
decode_token_latency_max_seconds
```

This helps identify occasional stalls caused by memory pressure, cache clearing, or synchronization.

### 3.3 Add benchmark presets

Add a new benchmark profile option:

```bash
gemma4 bench --profile single-user-latency
```

The profile should test:

```text
Short chat:
  prompt: 128 / 512 tokens
  decode: 64 / 128 tokens

Long prompt:
  prompt: 2048 / 8192 / 16384 tokens
  decode: 64 / 128 tokens

Repeated prefix:
  fixed prefix: 2048 / 8192 tokens
  varying suffix: 128 / 512 tokens
  decode: 64 tokens

Multi-turn simulation:
  turn 1: 512 prompt + 128 decode
  turn 2: append 256 tokens + 128 decode
  turn 3: append 256 tokens + 128 decode
```

### 3.4 Expand default benchmark matrix

Current benchmark defaults are too narrow. Add matrix coverage for:

```text
prefill_step_size:
  512, 1024, 2048, 4096, 8192

prefill_sync_policy:
  eval, async, none

prefill_cache_policy:
  clear, retain

decode_variant:
  custom
  custom_no_async
  custom_defer_ids
  custom_blockwise_8
  custom_blockwise_16
  custom_blockwise_32
  mlx_lm_generate_step
```

### Acceptance criteria

- Benchmark output clearly reports which prefill component is expensive.
- Benchmark output clearly reports decode sync overhead.
- Benchmark can compare `auto` against fixed chunk sizes.
- Benchmark can compare streaming-like and non-streaming-like decode paths.

---

## 4. Phase 1 — Prefill Chunk Size Optimization

### 4.1 Replace fixed `auto=512` with adaptive policy

Current `auto` behavior should be replaced with a dynamic policy.

Suggested initial implementation:

```python
def _prefill_step_size(value: PrefillStepSize, prompt_tokens: int) -> int:
    if value != "auto":
        return int(value)

    if prompt_tokens <= 1024:
        return 1024
    if prompt_tokens <= 8192:
        return 2048
    if prompt_tokens <= 32768:
        return 4096
    return 8192
```

Alternative more aggressive policy:

```python
if prompt_tokens <= 1024:
    return 1024
if prompt_tokens <= 4096:
    return 2048
if prompt_tokens <= 16384:
    return 4096
return 8192
```

The best policy should be selected by benchmark results.

### 4.2 Add memory-aware auto policy

Add optional memory-aware behavior:

```python
def _prefill_step_size_auto(prompt_tokens: int, active_memory_gb: float | None) -> int:
    ...
```

Rough strategy:

```text
If memory pressure is low:
  use larger chunks.

If memory pressure is high:
  reduce chunk size.

If OOM occurs:
  retry with smaller chunk size and cache clearing enabled.
```

### 4.3 Add OOM retry fallback

If prefill fails with memory-related error:

1. Clear MLX cache.
2. Reduce prefill chunk size by half.
3. Retry once or twice.
4. Emit warning in result.

Example fallback sequence:

```text
8192 -> 4096 -> 2048 -> 1024 -> 512
```

### Acceptance criteria

- `auto` is no longer always 512.
- Long-prompt prefill improves in benchmark.
- Short-prompt behavior remains stable.
- If memory is insufficient, engine falls back safely.

---

## 5. Phase 2 — Prefill Synchronization Optimization

### 5.1 Add `periodic` sync policy

Current policies:

```python
PrefillSyncPolicy = Literal["eval", "async", "none"]
```

Add:

```python
"periodic"
```

Suggested behavior:

```python
if policy == "periodic":
    if chunk_index % sync_every == 0 or is_last_chunk:
        mx.eval(states)
    else:
        mx.async_eval(states)
```

CLI/server option:

```bash
--prefill-sync-policy periodic
--prefill-sync-every 4
```

### 5.2 Change single-user default to `async` or `periodic`

For local single-user usage, default should not necessarily be `eval`.

Recommended default profile:

```text
prefill_sync_policy = async
```

or:

```text
prefill_sync_policy = periodic
prefill_sync_every = 4
```

### 5.3 Ensure final synchronization still happens

Even if chunk sync is `async` or `none`, ensure the final prefill token/logits are evaluated before timing prefill completion and before decode starts.

### Acceptance criteria

- Long-prompt prefill improves compared with per-chunk `eval`.
- Memory usage does not grow uncontrollably.
- Final logits/token are valid and deterministic for greedy decode.

---

## 6. Phase 3 — MLX Cache Clearing Optimization

### 6.1 Avoid unconditional per-chunk `mx.clear_cache()`

Current conservative behavior can clear MLX cache after every chunk.

For single-user local inference, this should be replaced by one of:

```text
retain
periodic clear
threshold-based clear
OOM retry clear
```

### 6.2 Add new cache policy options

Current:

```python
PrefillCachePolicy = Literal["clear", "retain"]
```

Add:

```python
"periodic"
"threshold"
```

Suggested behavior:

```python
if policy == "periodic":
    if chunk_index % clear_every == 0:
        mx.clear_cache()

if policy == "threshold":
    if mx.get_active_memory() > threshold_bytes:
        mx.clear_cache()
```

CLI/server options:

```bash
--prefill-cache-policy threshold
--prefill-cache-threshold-gb 18

--prefill-cache-policy periodic
--prefill-cache-clear-every 8
```

### 6.3 Change self-use default to `retain`

Recommended default for single-user speed profile:

```text
prefill_cache_policy = retain
```

Fallback strategy:

```text
If OOM occurs, retry with clear or smaller chunk size.
```

### Acceptance criteria

- `retain` or `threshold` improves prefill throughput.
- Engine still recovers from memory pressure.
- Benchmark reports clear-cache time separately.

---

## 7. Phase 4 — Session-Level KV Cache

This is likely the most valuable optimization for repeated local chat/agent usage.

### 7.1 Add session API

Add support for:

```json
{
  "session_id": "default",
  "prompt": "new user message",
  "max_tokens": 128,
  "append_to_session": true
}
```

The engine should maintain:

```python
@dataclass
class SessionState:
    token_ids: list[int]
    prompt_cache: list[object]
    generated_token_ids: list[int]
    last_access_time: float
```

Store in:

```python
self._sessions: OrderedDict[str, SessionState]
```

### 7.2 Only prefill newly appended tokens

For follow-up requests:

1. Retrieve session cache.
2. Format/encode only the new message or appended prompt.
3. Feed only new tokens into existing prompt cache.
4. Decode from the updated cache.
5. Append generated tokens to session state.

This avoids re-prefilling the whole chat history.

### 7.3 Add session management endpoints

Suggested HTTP endpoints:

```text
POST /generate
POST /session/reset
GET  /session/list
POST /session/clear
```

Minimal implementation can keep only `/generate` and accept:

```json
{
  "session_id": "main",
  "reset_session": false
}
```

### 7.4 Add CLI support

Example:

```bash
gemma4 serve --enable-sessions
```

Request:

```json
{
  "session_id": "main",
  "prompt": "Continue explaining this.",
  "max_tokens": 128
}
```

### 7.5 Handle prompt mismatch safely

If a session exists but incoming prompt does not match expected append-only behavior:

- Either reset session.
- Or fall back to normal full prefill.
- Emit a warning.

### 7.6 Session eviction

For single-user usage, keep it simple:

```python
max_sessions = 8
max_session_tokens = optional
max_total_session_cache_bytes = optional
```

Evict least recently used session when needed.

### Acceptance criteria

- Multi-turn chat no longer prefills full history every time.
- TTFT for follow-up turns improves significantly.
- Session reset works.
- Normal stateless generation remains supported.

---

## 8. Phase 5 — Prefix Cache Improvements

The current `cache_prefix` mechanism is useful but manual. Improve it for common local usage.

### 8.1 Add automatic longest-prefix cache matching

Current prefix cache requires an explicit `cache_prefix`.

Instead, when a prompt arrives:

1. Compare prompt token IDs against cached prefix token IDs.
2. Find the longest matching cached prefix.
3. Clone/fork that cache.
4. Only prefill the suffix.

Suggested method:

```python
def _find_longest_prefix_cache(self, prompt_ids: list[int]) -> PrefixCacheEntry | None:
    ...
```

### 8.2 Make prefix cache LRU

Current eviction is roughly FIFO. Replace with `OrderedDict` LRU behavior:

```python
self._prefix_cache.move_to_end(key)
self._prefix_cache.popitem(last=False)
```

### 8.3 Make prefix cache memory-aware

Do not limit only by entry count.

Add:

```python
max_prefix_cache_bytes: int | None
```

Estimate cache size using MLX array metadata when possible.

### 8.4 Measure prefix cache clone cost

Currently cache clone cost is mixed into prefill timing. Add separate stat:

```python
prefix_kv_cache_clone_seconds
```

This is important because cloning a long prefix KV cache may itself become expensive.

### Acceptance criteria

- Repeated system prompts and RAG templates are reused automatically.
- Prefix cache eviction is LRU.
- Prefix cache does not grow without memory control.
- Clone cost is visible in benchmark output.

---

## 9. Phase 6 — Blockwise Non-Streaming Decode

Streaming decode needs per-token sync. Non-streaming decode does not.

### 9.1 Add decode variants

Add:

```python
DecodeVariant = Literal[
    "custom",
    "custom_no_async",
    "custom_eval_next",
    "custom_defer_ids",
    "custom_blockwise_8",
    "custom_blockwise_16",
    "custom_blockwise_32",
    "mlx_lm_generate_step",
]
```

### 9.2 Implement blockwise decode

Pseudo-code:

```python
def decode_blockwise(block_size: int):
    generated = []
    token = first_token

    while len(generated) < max_tokens:
        token_arrays = []

        for _ in range(min(block_size, max_tokens - len(generated))):
            token_arrays.append(token)
            token = step(token)
            mx.async_eval(token)

        block_tokens = mx.concatenate(token_arrays, axis=0)
        mx.eval(block_tokens)
        token_ids = [int(x) for x in block_tokens.tolist()]

        for token_id in token_ids:
            if token_id in eos_token_ids:
                return generated
            generated.append(token_id)
```

### 9.3 Block size tradeoff

```text
block_size = 8:
  Better EOS responsiveness, lower over-generation risk.

block_size = 16:
  Good default candidate.

block_size = 32:
  Faster possible non-streaming decode, but more over-generation risk and larger lazy graph.
```

### 9.4 Separate streaming and non-streaming modes

Add request option:

```json
{
  "stream": false,
  "decode_variant": "custom_blockwise_16"
}
```

For streaming:

```text
Use custom one-token lookahead path.
```

For non-streaming:

```text
Use blockwise path.
```

### Acceptance criteria

- Non-streaming decode tok/s improves.
- EOS is checked at block boundaries.
- Streaming behavior remains unchanged.
- TTFT measurement remains meaningful for streaming mode.

---

## 10. Phase 7 — N-Gram Speculative Decoding

For single-user local decoding, speculative decoding can help when outputs contain repeated patterns or predictable text.

Start with n-gram speculative decoding because it does not require a draft model.

### 10.1 Implement prompt lookup / n-gram draft

Idea:

- Build an n-gram map from prompt tokens and generated tokens.
- When current suffix matches a previous n-gram prefix, propose the following tokens as draft.
- Verify several draft tokens with the target model in one pass.

### 10.2 Add decode variant

```python
"custom_speculative_ngram"
```

Request options:

```json
{
  "speculative_ngram_min": 3,
  "speculative_ngram_max": 6,
  "speculative_draft_tokens": 4
}
```

### 10.3 Verification logic

At each step:

1. Propose `k` draft tokens.
2. Run target model on those tokens using current cache.
3. Compare target greedy tokens with draft tokens.
4. Accept matching prefix.
5. On mismatch, accept target token and continue.

### 10.4 Best use cases

This helps most for:

- Code generation with repeated identifiers.
- Structured outputs.
- Repeated templates.
- Summaries that reuse source text.
- Agent logs or tool-call-like formats.

### Acceptance criteria

- Output must match greedy target model exactly.
- If no good n-gram draft exists, fallback overhead should be low.
- Benchmark reports acceptance rate.

---

## 11. Phase 8 — Optional Draft-Model Speculative Decoding

This is more complex than n-gram speculative decoding but can improve decode speed significantly if a small draft model is fast enough.

### 11.1 Add draft model loading

CLI/server options:

```bash
--draft-model /path/to/small/model
--speculative-draft-tokens 4
```

### 11.2 Add draft model state

Maintain separate draft model and draft KV cache.

### 11.3 Verify with target Gemma4

Use target model to verify draft tokens in blocks.

### 11.4 Risks

- More memory usage.
- More implementation complexity.
- Need compatible tokenizer.
- Draft model must be much faster than target.

### Recommendation

Do this only after:

1. Adaptive prefill is complete.
2. Session KV cache is complete.
3. Blockwise decode is complete.
4. Benchmarks show decode is still the main bottleneck.

---

## 12. Phase 9 — KV Cache Quantization and Sliding Window Improvements

### 12.1 Avoid globally disabling KV quantization

Current logic disables `kv_bits` for Gemma4 when shared-KV cache incompatibility is detected.

Better strategy:

```text
For cache entries that support `to_quantized`, quantize them.
For unsupported shared-KV entries, keep original precision.
```

This should be handled per cache entry rather than globally disabling KV quantization.

### 12.2 Add per-layer KV policy

Gemma4 has sliding-window attention. Ideally:

```text
Sliding-window layers:
  use ring buffer / capped KV size around sliding_window.

Global attention layers:
  preserve longer KV.

Shared-KV layers:
  use compatible non-quantized cache unless supported.
```

### 12.3 Add safer `max_kv_size` behavior

Global `max_kv_size` can hurt long-context quality if applied blindly.

Instead expose:

```text
--max-sliding-kv-size
--max-global-kv-size
```

If MLX cache internals do not expose per-layer control, leave this as a later low-level optimization.

### Acceptance criteria

- Long-context memory usage improves.
- Decode remains correct.
- KV quantization is applied where supported instead of being fully disabled.

---

## 13. Phase 10 — Interactive Local Mode

For single-user CLI usage, add an interactive mode to avoid repeated model load.

### 13.1 Add command

```bash
gemma4 chat --model /path/to/model
```

Behavior:

- Load model once.
- Keep `Gemma4Engine` alive.
- Keep session KV cache alive.
- User types messages interactively.

### 13.2 Benefits

- Avoid repeated model loading.
- Reuse session KV cache naturally.
- Better local experience than repeatedly calling `gemma4 infer`.

### Acceptance criteria

- Model loads once.
- Multi-turn chat uses session cache.
- `/reset` command clears current session.
- `/stats` shows last-run timing.

---

## 14. Recommended Implementation Order

### P0 — Must do first

1. Add detailed timing metrics.
2. Expand benchmark matrix.
3. Replace fixed `auto=512` with adaptive prefill chunk sizing.
4. Benchmark `retain` vs `clear` cache policy.
5. Benchmark `async` / `none` / `eval` sync policy.

Expected outcome:

- Clear visibility into actual prefill bottleneck.
- Likely immediate TTFT improvement on long prompts.

### P1 — Highest practical single-user value

1. Add session-level KV cache.
2. Add automatic prefix cache matching.
3. Make prefix cache LRU and memory-aware.
4. Add blockwise non-streaming decode variants.
5. Add interactive local chat mode.

Expected outcome:

- Much faster follow-up turns.
- Better local agent/chat workflow.
- Lower repeated prompt cost.

### P2 — Advanced decode acceleration

1. Add n-gram speculative decoding.
2. Add optional draft-model speculative decoding.
3. Try `mx.compile` on decode step as a benchmark variant.
4. Explore fused lm-head + argmax with custom MLX/Metal kernel.

Expected outcome:

- Faster long-output generation if decode remains the bottleneck.

### P3 — Deep engine internals

1. Per-layer sliding-window KV cache control.
2. Mixed KV quantization for supported cache entries.
3. Better cache forking / copy-on-write prefix cache.

Expected outcome:

- Lower memory pressure.
- Better long-context decode performance.
- Better scalability for long local sessions.

---

## 15. Suggested Default Single-User Profile

Add a profile named:

```text
single_user_fast
```

Suggested defaults:

```python
prefill_step_size = "auto"        # adaptive, not fixed 512
prefill_cache_policy = "retain"   # fallback to clear on OOM
prefill_sync_policy = "async"     # or periodic after testing
max_prefix_cache_entries = 8
max_sessions = 4
decode_variant = "custom"         # streaming default
non_stream_decode_variant = "custom_blockwise_16"
```

For long-prompt benchmarking:

```bash
gemma4 bench \
  --prompt-tokens 512,2048,8192,16384 \
  --decode-tokens 64,128 \
  --prefill-step-sizes 512,1024,2048,4096,8192 \
  --prefill-sync-policies eval,async,none \
  --prefill-cache-policy both \
  --decode-variants custom,custom_blockwise_8,custom_blockwise_16,custom_blockwise_32,mlx_lm_generate_step
```

---

## 16. Agent Task List

### Task 1 — Add detailed timing breakdown

Files likely involved:

- `stats.py`
- `inference.py`
- `benchmark.py`

Requirements:

- Extend `RunStats` with optional detailed timing fields.
- Measure encode time.
- Measure prefix cache lookup/build/clone time.
- Measure prefill model time.
- Measure prefill sync time.
- Measure MLX clear-cache time.
- Measure decode sync and token extraction time.
- Include new fields in JSON output.
- Include important fields in benchmark summary.

### Task 2 — Adaptive prefill chunk size

Files likely involved:

- `inference.py`
- `cli.py`
- `benchmark.py`

Requirements:

- Replace fixed `auto=512` with adaptive policy.
- Add benchmark matrix for fixed chunk sizes.
- Add OOM fallback if feasible.
- Keep existing explicit chunk-size behavior unchanged.

### Task 3 — Better prefill sync/cache policies

Files likely involved:

- `inference.py`
- `cli.py`
- `server.py`
- `benchmark.py`

Requirements:

- Add `periodic` sync policy.
- Add `threshold` or `periodic` cache-clear policy.
- Add CLI/server arguments for policy parameters.
- Ensure final prefill result is synchronized.
- Benchmark against existing defaults.

### Task 4 — Session KV cache

Files likely involved:

- `inference.py`
- `server.py`
- `cli.py`

Requirements:

- Add `SessionState` dataclass.
- Add `self._sessions` to `Gemma4Engine`.
- Add `session_id`, `reset_session`, and `append_to_session` support.
- Reuse KV cache for follow-up turns.
- Add LRU eviction.
- Add stats showing session cache hit/miss and reused token count.

### Task 5 — Automatic prefix cache matching

Files likely involved:

- `inference.py`

Requirements:

- Implement longest-prefix cache matching.
- Convert prefix cache to LRU.
- Add memory-aware cache limit if feasible.
- Measure prefix cache clone cost.

### Task 6 — Blockwise non-streaming decode

Files likely involved:

- `inference.py`
- `benchmark.py`
- `cli.py`

Requirements:

- Add `custom_blockwise_8`, `custom_blockwise_16`, `custom_blockwise_32` variants.
- Implement blockwise sync.
- Check EOS at block boundaries.
- Keep streaming path unchanged.
- Benchmark against `custom` and `custom_defer_ids`.

### Task 7 — Interactive CLI mode

Files likely involved:

- `cli.py`
- `inference.py`

Requirements:

- Add `gemma4 chat` command.
- Load model once.
- Keep session cache alive.
- Support `/reset`, `/exit`, `/stats`.

### Task 8 — N-gram speculative decoding

Files likely involved:

- `inference.py`
- `benchmark.py`
- `cli.py`

Requirements:

- Add n-gram draft proposal.
- Add target verification.
- Guarantee greedy-equivalent output.
- Report acceptance rate.
- Benchmark on code/text/template prompts.

---

## 17. Success Metrics

Track these before and after each phase:

### Prefill metrics

```text
prefill_tokens_per_second
time_to_first_token_seconds
prefill_model_seconds
prefill_sync_seconds
prefill_clear_cache_seconds
prefix_cache_clone_seconds
```

### Decode metrics

```text
decode_tokens_per_second
decode_token_latency_p50_seconds
decode_token_latency_p95_seconds
decode_sync_seconds
decode_token_item_seconds
```

### Cache metrics

```text
prefix_cache_hit
prefix_tokens_reused
session_cache_hit
session_tokens_reused
prefix_cache_entries
session_count
active_memory_gb
peak_memory_gb
cache_memory_gb
```

### Correctness metrics

```text
tokens_match_baseline
generated_token_hash
EOS handling correctness
no crash under long prompt
OOM fallback works
```

---

## 18. Expected Impact

### High-confidence improvements

- Adaptive prefill chunk size should improve long-prompt prefill.
- Avoiding per-chunk `mx.clear_cache()` should improve prefill when memory allows.
- Avoiding per-chunk blocking `mx.eval()` should improve prefill throughput.
- Session KV cache should greatly improve follow-up turns.
- Persistent server/interactive mode avoids repeated model loading.

### Medium-confidence improvements

- Blockwise non-streaming decode should improve non-streaming decode throughput.
- Automatic prefix cache matching should improve repeated system/RAG prompt workloads.
- N-gram speculative decoding should help structured/repetitive outputs.

### Lower-confidence / advanced improvements

- `mx.compile` decode step may help, but cache mutation may complicate it.
- Custom fused argmax/lm-head kernel may help but requires deeper MLX/Metal work.
- Per-layer KV cache control may require internal MLX cache changes.

---

## 19. What Not to Prioritize for Single-User Usage

Do not spend early engineering effort on:

- Continuous batching.
- Multi-user request scheduling.
- Full async server rewrite.
- Complex multi-tenant queueing.
- Throughput optimization for many simultaneous users.

These are useful for deployment, but not the main bottleneck for local single-user usage.

---

## 20. Final Recommended Roadmap

For the next implementation cycle, do this exact order:

```text
1. Add detailed timing breakdown.
2. Implement adaptive prefill auto chunk size.
3. Benchmark retain/async/none/large chunks.
4. Change single-user default profile based on benchmark results.
5. Implement session KV cache.
6. Implement blockwise non-streaming decode.
7. Implement automatic prefix cache matching.
8. Add interactive CLI mode.
9. Consider n-gram speculative decoding.
10. Only then consider low-level MLX/Metal or draft-model speculative decoding.
```

This order should maximize real single-user speed improvements while keeping engineering risk manageable.
