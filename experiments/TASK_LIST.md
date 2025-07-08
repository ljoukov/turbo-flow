# o3-deep-research – TASK LIST

For each item below, the browsing agent should (1) collect authoritative sources (papers, GitHub repos, official docs, blog posts), (2) extract the key facts/code snippets/config files, and (3) summarise findings in a short written brief plus links. Unless stated otherwise, focus on the most recent (≤ 5 years) material.

---

## PHASE-1 Data Engine – High-fidelity SRP simulations

1. NASA / SpaceX Falcon-9 SRP public dataset  
   • Locate the full experimental data package (pressure taps, geometry, test conditions, reports).  
   • Identify: file formats, coordinate systems, licensing, any prior digital reconstructions.  
   Deliverable: download links + data dictionary + CSV of tap locations.

2. State-of-the-art SRP CFD literature survey  
   • Collect LES / WMLES / DG papers on supersonic retro-propulsion (Mars EDL, Falcon-9, etc.).  
   • Extract recommended boundary conditions, grid resolutions, turbulence models, plume inlet treatments.  
   Deliverable: annotated bibliography + table of “best-practice” solver settings.

3. SU2 capability check for SRP  
   • Verify SU2 support for: ­Discontinuous Galerkin, WMLES, reacting plume option, multi-zone meshes.  
   • Find example _.cfg files or tutorials close to SRP or high-Mach jet-into-cross-flow.  
   Deliverable: list of SU2 branches/features required + minimal working _.cfg template.

4. Unstructured mesh generation tools that integrate with SU2 for rocket bodies  
   • Compare Pointwise, Gmsh, pyHyp, cfMesh.  
   • Identify automation options (Python APIs, scripting).  
   Deliverable: recommendation with pros/cons and sample script for parametric Falcon-like body.

5. Sobol / Latin-Hypercube DOE libraries in Python  
   • Find fast, well-maintained packages that return reproducible, scrambled Sobol sequences.  
   Deliverable: code snippet generating 1 000 samples in given parameter ranges.

6. HPC cluster benchmarks for SU2 WMLES jobs  
   • Gather runtime vs. node-count scaling numbers on CPUs and GPUs (if available) for similar mesh sizes (10–50 M cells).  
   Deliverable: table of typical core-hours & memory to plan scheduling.

---

## PHASE-2 Model Implementation & Training

7. JAX-native Graph Neural Network frameworks  
   • Survey jraph, GraphNets (DeepMind), Haiku, Equinox, Objax, Graph-Tau.  
   • Assess feature completeness: message-passing, batching, sparse ops, distributed training.  
   Deliverable: comparison matrix + recommended stack + installation instructions.

8. Physics-Informed Neural Operator (PINO) & graph-based PDE research  
   • Collect papers/repos using graph/PINO for CFD (e.g., “Graph Neural Operator”, “Phiflow-PINN-Mesh”, “MGNO”).  
   • Extract loss formulation and implementation details.  
   Deliverable: code excerpts for Navier-Stokes residual on unstructured graph in JAX.

9. Automatic differentiation of spatial derivatives on irregular meshes in JAX  
   • Look for libraries or examples computing ∇·F, Laplacian, etc., with scatter-gather ops.  
   Deliverable: validated JAX snippet for first-order derivatives on node‐based data.

10. Large-scale JAX training on multi-GPU / multi-host  
    • Best practices for jax.pmap / jax.distributed, gradient accumulation, mixed precision.  
    • Case studies reporting throughput on A100/H100.  
    Deliverable: checklist + reference scripts.

11. Inference speed optimisation for GNNs in JAX  
    • Techniques: XLA compilation caching, jitted static shapes, sparse matmul accelerators, quantisation.  
    Deliverable: actionable guidelines expected to keep 3-D inference < 10 min on single A100.

12. Efficient graph data storage & loading  
    • Benchmark HDF5, TFRecord, NPZ, Parquet for million-node meshes.  
    • Identify existing Python loaders that zero-copy into JAX DeviceArrays.  
    Deliverable: recommendation + code snippet.

13. Pre-training with low-fidelity RANS data  
    • Locate open RANS rocket plume datasets or fast public solvers (e.g., FUN3D-RANS, OpenFOAM).  
    • Estimate mesh sizes vs. turnaround time to produce 10 k cases.  
    Deliverable: source list + estimated compute cost.

---

## PHASE-3 Validation & Deployment

14. Interpolation of predicted fields onto experimental tap locations  
    • Survey Python libraries for nearest-neighbour / inverse-distance on unstructured 3-D meshes (PyVista, SciPy KDTree, vtk).  
    Deliverable: benchmark of methods + preferred implementation.

15. Statistical metrics & uncertainty bands for aerodynamic coefficients  
    • Best practices in CFD/experimental comparison: L2 error, Cp strip averages, confidence intervals.  
    Deliverable: short guideline + formulae.

16. Packaging JAX models for easy inference  
    • Investigate Orbax, Flax checkpoints, and ONNX export status for JAX.  
    Deliverable: recommended format + minimal “predict.py”.

17. Licensing & export-control audit  
    • Identify any ITAR/EAR restrictions on SRP data or trained model weights.  
    Deliverable: memo summarising legal considerations.

---

## CROSS-CUTTING (Tools & DevOps)

18. Mesh/flow post-processing Python ecosystem  
    • Latest versions of meshio, PyVista, VTK-Python, h5py with GPU-direct support.  
    Deliverable: compatibility matrix with JAX & CUDA 12.

19. Experiment tracking & visualization  
    • Compare Weights-and-Biases, TensorBoard, Comet for large-scale JAX jobs.  
    Deliverable: recommendation + setup steps.

20. Continuous integration templates for HPC + cloud (GitHub Actions, GitLab CI)  
    • Examples that build SU2, run JAX unit tests, and archive artifacts.  
    Deliverable: sample YAML workflows.

---

## FORM OF THE REPORT

For every numbered task, o3-deep-research should return:  
• Executive summary (≤ 200 words).  
• Bullet list of key findings.  
• Links to sources (DOI, arXiv, GitHub, docs).  
• Code/config snippets where applicable.

End of task list.
