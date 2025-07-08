# Turbo-Flow – Laptop-Scale Experiments

Each experiment is self-contained and feasible on a Mac laptop (Apple Silicon or Intel) within ≤ 30 minutes wall-clock.  
Follow the same pattern: **clone / create a fresh conda env → run the script(s) in the corresponding folder**.  
Results from successful experiments feed directly into the large-scale phases in **PLAN.md**.

---

### Experiment 1 – 2-D Supersonic Cylinder (SU2 ➜ Toy GNN)

**Goal** Generate a miniature CFD dataset (5 cases) and train a one-layer GNN that predicts surface Cp with < 15 % L2 error.  
**Why it matters** Validates the end-to-end _data‐engine → model_ loop at tiny scale.  
**Setup Time** 5 min  **Run Time** ≈ 10 min  
**Compute** CPU-only; optional Apple-GPU with JAX-Metal or MLX.  
**Key Tools** SU2 v8, meshio, h5py, JAX or MLX, Matplotlib  
**Success** Mean L2(Cp) < 15 % on all 5 AoA cases.  
**Next Step** Scale recipes for Phase 1 simulation post-processing.

---

### Experiment 2 – Method-of-Manufactured-Solutions (MMS) Residual Sanity Check

**Goal** Verify the physics-informed loss implementation by driving analytical MMS residuals to < 1 e-4.  
**Why it matters** Guarantees automatic-differentiation residuals are coded correctly before expensive training.  
**Setup Time** 3 min  **Run Time** < 2 min  
**Compute** CPU-only  
**Key Tools** JAX/MLX, SymPy, NumPy  
**Success** Residual MSE < 1 e-4 within 500 optimisation steps.  
**Next Step** Integrate loss component into full model (Phase 2).

---

### Experiment 3 – Mesh-Loader & Graph-Builder Benchmark

**Goal** Measure I/O latency and memory footprint when converting a 100 k-cell SU2 mesh to graph tensors.  
**Why it matters** Ensures the data pipeline will not bottleneck distributed training.  
**Setup Time** 2 min  **Run Time** < 1 min  
**Compute** CPU-only  
**Key Tools** meshio, h5py, pandas, Memory-profiler  
**Success** Conversion < 3 s, peak RAM < 1 GB.  
**Next Step** Adopt findings for the large dataset builder script.

---

### Experiment 4 – Message-Passing Throughput Micro-Bench (JAX vs MLX)

**Goal** Benchmark messages/s for a single GNN layer on identical random graphs across JAX-CPU, JAX-Metal, and MLX back-ends.  
**Why it matters** Choose the fastest Apple-Silicon-friendly framework before full implementation.  
**Setup Time** 4 min  **Run Time** < 3 min  
**Compute** Apple-GPU optional (Metal); falls back to CPU.  
**Key Tools** JAX, MLX, timeit, matplotlib  
**Success** Report ops/s & latency; pick winner for Phase 2.  
**Next Step** Lock framework choice for main architecture.

---

### Experiment 5 – Tiny 3-D RANS Plume Case (End-to-End)

**Goal** Run a 25 k-cell RANS plume simulation, post-process to HDF5, visualise variables to confirm correctness.  
**Why it matters** Exercise 3-D workflow and uncover SU2 quirks early.  
**Setup Time** 10 min  **Run Time** ≈ 10 min  
**Compute** CPU-only; requires ~8 GB RAM.  
**Key Tools** SU2, Paraview (optional), meshio, h5py  
**Success** HDF5 file with validated fields; visual match to Paraview screenshot.  
**Next Step** Template for large Phase 1 WMLES jobs.

---

### Experiment 6 – Inference Precision Sweep on Apple GPU

**Goal** Profile inference latency for fp32, bf16, and fp16 on an Apple-Silicon GPU using a frozen toy operator (from Exp 1).  
**Why it matters** Quantifies realistic speed-ups and guides precision policy for Phase 3 deployment.  
**Setup Time** 4 min  **Run Time** < 5 min  
**Compute** Apple-GPU (Metal) required; falls back to CPU with warnings.  
**Key Tools** JAX-Metal or MLX, timeit, numpy  
**Success** Plot latency vs precision; document speed & accuracy delta.  
**Next Step** Feed precision choice into final performance target (< 10 min inference).

---

## Getting Started

1. `git clone https://github.com/ljoukov/turbo-flow.git && cd turbo-flow`
2. Pick an experiment folder (will be created as we go) and follow its **README.md**.
3. Use a fresh `conda env create -f environment.yml` per experiment to avoid library clashes.
