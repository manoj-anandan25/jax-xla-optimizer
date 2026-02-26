<h1 align="center"> JAX / XLA Whole-Program Optimizer</h1>
<p align="center">
Compiler-Level ML Optimization with JIT Fusion
</p>

<p align="center">
  <img src="https://img.shields.io/badge/JAX-0.7.2-9cf?style=for-the-badge&logo=google">
  <img src="https://img.shields.io/badge/XLA-Compiler-blue?style=for-the-badge">
  <img src="https://img.shields.io/badge/Paradigm-Functional-success?style=for-the-badge">
</p>

##  Problem Statement
Standard Python-based deep learning loops execute operations sequentially in "Eager Mode." This leads to:
- **Interpreter Bottlenecks**: High dispatch overhead from the Python interpreter.
- **Memory Round-trips**: Intermediate tensors are written to and read from VRAM between every operation.
- **Sub-optimal Graphs**: Lack of cross-operation optimization (e.g., fusing an activation function with a dot product).

**Goal:** Transform the entire training step into a single, fused, hardware-specific binary via XLA compilation.

##  Overview
This project utilizes **JAX** and the **XLA (Accelerated Linear Algebra)** compiler to optimize a training pipeline. By shifting from imperative execution to a compiled graph, we eliminate dispatch overhead and maximize GPU utilization through kernel fusion.

##  Technical Architecture & Compiler Pipeline
The project follows the functional transformation pipeline:
1. **Python Function**: Defined using pure functions and immutable state.
2. **JAX Tracing**: The function is traced into a **JAXPR** (JAX Intermediate Representation).
3. **XLA Lowering**: JAXPR is lowered to **HLO (High-Level Optimizer)** IR.
4. **HLO Fusion**: The compiler fuses multiple mathematical ops into a single optimized GPU executable.



<p align="center">
  <img width="800" height="1024" alt="architecture-daigram" src="https://github.com/user-attachments/assets/88ddfd78-a938-4935-9f9a-1d68597729f1"/>
</p>

##  Performance Benchmarks (NVIDIA T4)
By using `@jax.jit`, the entire forward pass, loss calculation, and gradient update are optimized into a single GPU call.

| Mode | Step Latency (ms) | Speedup |
| :--- | :--- | :--- |
| Eager Execution (Simulated) | ~2.10 ms | 1.00× |
| **JIT-Compiled Step** | **0.204 ms** | **~10.3×** |

> **Note**: The **0.204 ms** latency includes the full `value_and_grad` training step on a 128x128 input dimension.

##  Core Engineering Highlights
- **Stateless Training**: Parameters are managed as **PyTrees**, enabling pure function transformations.
- **Whole-Program JIT**: Unlike operation-level JIT, this compiles the *entire* training step into one kernel.
- **Algebraic Simplification**: XLA identifies and eliminates redundant operations during the lowering phase.

##  Future Improvements
- **Data Parallelism**: Scaling to multiple GPUs using `jax.pmap`.
- **Custom Pallas Kernels**: Hand-writing GPU kernels within the JAX ecosystem.
- **Mixed Precision**: Implementing `float16` training for higher throughput.

##  Build & Run
```bash
pip install -r requirements.txt
python benchmark.py
