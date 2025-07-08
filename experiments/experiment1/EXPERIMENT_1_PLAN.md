# Experiment 1 – 2-D Supersonic Cylinder (Toy Pipeline)

This document is the hands-on recipe for running **Experiment 1** on a Mac laptop (Intel or Apple-Silicon).  
It walks from environment creation to success metrics in ≈ 15 minutes.

---

## 1. Objective

Create a **miniature CFD-to-GNN** workflow:

1. Run five 2-D SU2 simulations of Mach 2 flow over a cylinder at small angles of attack.
2. Convert each case into a graph data file (`HDF5`).
3. Train a _one-layer_ GNN (JAX **or** MLX) to predict surface coefficient of pressure **Cp**.
4. Achieve **< 15 % L2 error** on all five cases.

Why? This validates the full Turbo-Flow toolchain at laptop scale before scaling to 3-D WMLES.

---

## 2. Hardware / Software Matrix

| Item | Minimum            | Notes                                  |
| ---- | ------------------ | -------------------------------------- |
| CPU  | 4 cores            | Intel or Apple                         |
| RAM  | 8 GB               | Peak ~2 GB                             |
| GPU  | Optional Apple-GPU | JAX-Metal or MLX speeds up training ×3 |
| OS   | macOS 12+          | Tested on macOS 14                     |

Two framework options are supported:

| Framework     | Pros                                                                      | Install Hint                                              |
| ------------- | ------------------------------------------------------------------------- | --------------------------------------------------------- |
| **JAX-Metal** | Mature API, same code as future HPC cluster, Metal backend uses Apple GPU | `pip install jax-metal==0.0.6 jaxlib==0.4.28 jax==0.4.28` |
| **MLX**       | Apple-native, streamlined for Metal, easier mixed precision               | `pip install mlx` (arm64 only)                            |

Choose one; code paths are abstracted behind `backend.py`.

---

## 3. Step-By-Step

### 3.1 Clone & Create Conda Environment

```bash
git clone https://github.com/ljoukov/turbo-flow.git
cd turbo-flow

# create env file on-the-fly
cat <<'EOF' > environment.yml
name: turbo_exp1
channels:  [conda-forge]
dependencies:
  - python=3.11
  - meshio
  - h5py
  - matplotlib
  - numpy
  - tqdm
  - pip
  - pip:
      - su2-gui==8.0.0
      # choose ONE of the following blocks
      # JAX-Metal (Intel or Apple Silicon):
      - jax-metal==0.0.6
      - jaxlib==0.4.28
      - jax==0.4.28
      # --- OR ---
      # MLX (Apple Silicon only):
      # - mlx
EOF

conda env create -f environment.yml
conda activate turbo_exp1
```

### 3.2 Generate SU2 Cases (≈ 2 min)

A helper script writes five config files and runs SU2 sequentially.

```bash
python scripts/exp1_make_cases.py   # creates ./exp1/case_<AoA>/config.cfg
python scripts/exp1_run_su2.py      # driver; prints progress bar
```

Each run solves ∼1 k-cell O-mesh in < 20 s CPU.

### 3.3 Post-Process to Graph HDF5 (≈ 2 min)

```bash
python scripts/exp1_postprocess.py  --root ./exp1  --out ./exp1/data
# outputs: case0.h5 … case4.h5
```

Inside each HDF5:

```
/coords        (N,3) float32
/edges         (E,2) int32
/node_features (N,K) float32
/cp_surface    (M,)  float32     # target for learning
```

### 3.4 Minimal GNN Definition

`models/minigcn.py` implements:

```
Encoder: MLP(d_in → 64)
MessagePassing: edge_mlp(128→64) + aggregation(sum) + update_mlp(128→64)
Decoder: MLP(64→1) ⇒ Cp
```

Backend-agnostic via `backend.py` (`jax.numpy` or `mlx.core`).

### 3.5 Train (≈ 3 min CPU, 1 min GPU)

```bash
python train.py \
    --data ./exp1/data \
    --backend jax      # or mlx
```

Hyper-parameters:

```
epochs       300
batch_size   1 (graph per step)
lr           1e-3 (Adam)
scheduler    cosine
```

### 3.6 Evaluate

The script prints:

```
AoA  L2[%]
0    12.1
1    13.5
...
```

Success if **mean L2 < 15 %**.  
It also saves `cp_comparison.png` with analytic vs predicted Cp curves.

---

## 4. Clean-Up

To free space:

```bash
rm -rf exp1/case_*/*.csv  exp1/data/*.h5
```

---

## 5. Expected Timeline

| Task      | Time       |
| --------- | ---------- |
| Env setup | 5 min      |
| SU2 runs  | 2 min      |
| Post-proc | 2 min      |
| Training  | 3 min      |
| Analysis  | 3 min      |
| **Total** | **15 min** |

---

## 6. What We Learn

- Verifies SU2 + meshio interoperability on macOS.
- Confirms backend (JAX-Metal **or** MLX) runs graph ML on Apple GPU.
- Provides code templates reused for Phase 1 data engine and Phase 2 model.
