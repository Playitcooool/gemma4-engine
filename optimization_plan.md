# Gemma 4 MLX Inference Engine: Optimization Plan

This document outlines a detailed, phase-by-phase engineering plan to optimize the Gemma 4 MLX inference engine. The goal is to maximize prompt prefill speed, autoregressive decoding throughput, and memory efficiency on Apple Silicon hardware.

---

## Phase 1: Eliminate CPU-GPU Memory Transfer Barriers

The custom backend `RustMetalBackend` currently forces memory transfer between GPU and CPU at every token decoding step, introducing significant latency.

### Action Items
1. **Deprecate CPU-based Argmax in Custom Backend**:
   - In [lib.rs](file:///Volumes/Samsung/Projects/gemma4-engine/crates/gemma4-kernels/src/lib.rs#L47-L50), the `greedy_argmax` function should compile and call the defined Metal shader rather than falling back to `cpu_argmax` via `numpy` CPU array slice.
2. **Implement Zero-Copy / Direct MLX Memory Binding**:
   - Instead of copying the MLX array to NumPy and then passing it to Rust (which copies it to CPU RAM), pass raw memory buffers directly. If possible, retrieve the underlying unified memory buffer from the MLX array using `data_pointer()` and pass it to Rust/Metal via PyO3.
3. **Rewrite the Metal Kernel for Parallel Reduction**:
   - Modify `greedy_argmax` in [lib.rs](file:///Volumes/Samsung/Projects/gemma4-engine/crates/gemma4-kernels/src/lib.rs#L9-L28) to run in parallel. The current kernel uses a single thread (`tid == 0`) with a sequential loop over all vocabulary logits.
   - Implement block-level/warp-level parallel reduction using threadgroup memory to find the argmax.

---

## Phase 2: Autoregressive Decoding Compilation (`mx.compile`)

Dynamic graph building in Python during decoding incurs high runtime overhead. Compiling the step execution graph fuses operations into a single Metal kernel launch.

### Action Items
1. **Compile the Decode Step Function**:
   - In [inference.py](file:///Volumes/Samsung/Projects/gemma4-engine/src/gemma4_engine/inference.py#L554-L557), compile the `step` function:
     ```python
     compiled_step = mx.compile(step)
     ```
2. **Ensure Compilation Cache Reuse**:
   - Ensure the compiled function does not trigger recompilations (which occurs if array shapes or types change dynamically across steps).

---

## Phase 3: Prefill Pipelining and Async Synchronization

The prefill phase chunking incurs CPU-GPU synchronization stalls and allocator thrashing.

### Action Items
1. **Transition to Asynchronous Evaluation**:
   - Avoid executing `mx.eval` on the prompt cache states after every single prefill chunk in [inference.py](file:///Volumes/Samsung/Projects/gemma4-engine/src/gemma4_engine/inference.py#L565-L566).
   - Use `mx.async_eval` or queue the evaluations to the stream, performing a single blocking synchronization (`mx.eval`) only on the final chunk.
2. **Minimize Allocator Cache Clearing**:
   - Update the default `--prefill-cache-policy` from `clear` to `retain` in [backends.py](file:///Volumes/Samsung/Projects/gemma4-engine/src/gemma4_engine/backends.py) / [cli.py](file:///Volumes/Samsung/Projects/gemma4-engine/src/gemma4_engine/cli.py#L129). Calling `mx.clear_cache()` inside the chunk loop forces constant deallocations and reallocations.

---

## Phase 4: Zero-Copy Prefix Cache Cloning

Prefix cache reuse is slowed down by cloning operations that deep-copy underlying array states.

### Action Items
1. **Bypass `copy.deepcopy` for MLX Arrays**:
   - In `_clone_prompt_cache` in [inference.py](file:///Volumes/Samsung/Projects/gemma4-engine/src/gemma4_engine/inference.py#L292-L298), perform shallow copying of metadata and call MLX's native copy (`mx.copy()` or slicing) on the actual array states. This will eliminate heavy Python object traversal and reflection overhead.

---

## Phase 5: Integrate Prefix Caching with Speculative Decoding

Currently, speculative decoding cannot utilize the prefill prefix cache, limiting its speed on chat or agentic loops.

### Action Items
1. **Wire Prefix Cache into target/draft models**:
   - Modify `SpeculativeRuntime.generate` in [speculative.py](file:///Volumes/Samsung/Projects/gemma4-engine/src/gemma4_engine/speculative.py#L31-L90) to accept a target model prefix cache.
   - Adapt the draft model (drafter) to also leverage a matching prefix cache so that prefill computation is avoided for both models when a cache hit occurs.

---

## Phase 6: Disk Cache Eviction Strategy

Currently, tokenized prefix outputs grow indefinitely on disk, which could eventually fill up the user's storage.

### Action Items
1. **Implement LRU Disk Pruning**:
   - Enhance the `HierarchicalTokenCache` in [token_cache.py](file:///Volumes/Samsung/Projects/gemma4-engine/src/gemma4_engine/token_cache.py) with a maximum disk storage limit (e.g. 500 MB).
   - Track cache access times and delete the least recently used `.g4tokens` files when the disk folder exceeds the threshold.
