use numpy::PyReadonlyArrayDyn;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

const GREEDY_ARGMAX_METAL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void greedy_argmax(
    device const float* logits [[buffer(0)]],
    device uint* out_index [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid != 0 || n == 0) {
        return;
    }
    float best = logits[0];
    uint best_index = 0;
    for (uint i = 1; i < n; i++) {
        float value = logits[i];
        if (value > best) {
            best = value;
            best_index = i;
        }
    }
    out_index[0] = best_index;
}
"#;

fn cpu_argmax(values: &[f32]) -> PyResult<usize> {
    if values.is_empty() {
        return Err(PyValueError::new_err("greedy_argmax requires at least one value"));
    }
    let mut best_index = 0usize;
    let mut best_value = values[0];
    for (index, value) in values.iter().copied().enumerate().skip(1) {
        if value > best_value {
            best_value = value;
            best_index = index;
        }
    }
    Ok(best_index)
}

#[pyfunction]
fn greedy_argmax(logits: PyReadonlyArrayDyn<f32>) -> PyResult<usize> {
    let slice = logits.as_slice()?;
    cpu_argmax(slice)
}

#[pyfunction]
fn metal_source() -> &'static str {
    GREEDY_ARGMAX_METAL
}

#[pymodule]
fn gemma4_kernels(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(greedy_argmax, module)?)?;
    module.add_function(wrap_pyfunction!(metal_source, module)?)?;
    Ok(())
}
