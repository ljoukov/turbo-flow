# PLAN.md: Project Turbo-Flow

## 1. Mission

To build, train, and validate a neural network model that accurately simulates supersonic retro-propulsion (SRP) flow fields orders of magnitude faster than traditional high-fidelity methods. The model will be validated against the public NASA/SpaceX Falcon 9 dataset.

## 2. Technical Objective: The "Triple Crown"

1.  **ACCURACY:** Achieve **< 5% L2 error** on surface pressure (`Cp`) compared to experimental data for the Falcon 9 SRP validation case.
2.  **SPEED:** Inference time for a full 3D flow field must be **< 10 minutes** on a single NVIDIA A100/H100 GPU.
3.  **GENERALIZATION:** The single trained model must maintain **< 10% error** on unseen test cases varying in Mach number, altitude, and throttle setting.

## 3. Core Architecture: Physics-Informed Graph Neural Operator

We will implement a neural operator based on a Graph Neural Network (GNN) that learns the mapping from physical conditions to a steady-state flow solution on an unstructured mesh.

- **Framework:** **JAX** with the **JAX MD** or a custom GNN library for message passing.
- **Input to Model:**
  1.  A graph representing the simulation mesh:
      - `nodes`: `(N, 3)` tensor of node coordinates (x, y, z).
      - `edges`: `(E, 2)` tensor of connections between nodes.
  2.  Boundary condition information encoded as node/edge features (e.g., one-hot encoding for inlet, outlet, wall, symmetry, plume inlet).
  3.  Global physical parameters: `(mach_freestream, reynolds_number, plume_pressure_ratio, angle_of_attack)`.
- **Output of Model:** A `(N, 5)` tensor representing the conserved flow variables (`ρ, ρu, ρv, ρw, E`) at each of the `N` mesh nodes.

## 4. Execution Plan: A Three-Phase Technical Sprint

### Phase 1: Data Engine - Simulation & Processing

This phase is dedicated to creating the fuel for our model: a massive, multi-parameter dataset of high-fidelity simulations.

**Task 1.1: Simulation Generation**

- **Solver:** Utilize a high-order Discontinuous Galerkin (DG) or Wall-Modeled LES (WMLES) solver via the **SU2** open-source suite. SU2 is well-validated for aerospace applications.
- **Geometry:** A parameterized CAD model of a generic rocket body similar to the Falcon 9.
- **Simulation Matrix:** Generate a Sobol sequence or Latin Hypercube sample of ~1,000 parameter sets to ensure efficient coverage of the design space:
  - `Mach`: [1.5, 4.0]
  - `Altitude (via Reynolds #)`: [20km, 50km]
  - `Throttle (via Plume NPR)`: [20, 200]
  - `Angle of Attack`: [0, 5] degrees
- **Execution:** Script and deploy simulation jobs on a large HPC cluster.

**Task 1.2: Data Pipeline**

- **Scripting:** Develop Python scripts using `meshio` and `h5py` to automate post-processing.
- **Process:** For each completed simulation case:
  1.  Read the unstructured mesh file (`.su2`, `.cgns`).
  2.  Read the solution file (`.vtu`, `.dat`).
  3.  Extract node coordinates, edge connectivity, and flow variables.
  4.  Extract boundary condition flags for each node/edge.
  5.  Normalize all physical quantities.
  6.  Save the processed data into a standardized `HDF5` or `TFRecord` format for efficient loading during training. Each file will contain the graph structure and the corresponding solution.
- **Final Dataset:** A curated collection of ~1,000 processed data files, split into 80% training, 10% validation, and 10% held-out test sets.

---

### Phase 2: Model Implementation & Training

This phase focuses on building and training the `Turbo-Flow` model itself.

**Task 2.1: Model Implementation (JAX)**

- **GNN Core:** Implement a `MessagePassing` GNN architecture. This involves three key learnable functions (Multi-Layer Perceptrons - MLPs):
  1.  `edge_function`: Computes messages based on the state of connected nodes and edge attributes.
  2.  `node_function`: Aggregates incoming messages and updates the node state.
  3.  `global_function`: Integrates information from all nodes to update global context (if needed).
- **Encoder/Decoder:**
  - An `Encoder` MLP that processes input node features and global parameters into a high-dimensional latent representation.
  - A `Processor` block composed of 10-15 stacked GNN layers to iteratively refine the solution.
  - A `Decoder` MLP that maps the final latent node representations back to the physical flow variables.
- **Code Structure:** A modular design with clear separation for data loading, model definition, and training loops.

**Task 2.2: Physics-Informed Loss Function**

- The total loss will be a weighted sum: `L_total = w_data * L_data + w_phys * L_phys`.
- **Data Loss (`L_data`):** Standard Mean Squared Error (MSE) between the model's predicted flow variables and the ground truth from our simulation data.
  `L_data = mean((y_pred - y_true)^2)`
- **Physics Loss (`L_phys`):** A "soft constraint" loss that penalizes violations of the Navier-Stokes equations in their differential form.
  1.  Use JAX's automatic differentiation (`jax.grad`) to compute spatial derivatives of the network's output (`∂u/∂x`, `∂p/∂y`, etc.) directly on the graph.
  2.  Plug these derivatives into the residual form of the conservation equations (mass, momentum, energy).
  3.  `L_phys` is the MSE of these residuals, driving them towards zero across the entire domain.

**Task 2.3: Training Protocol**

- **Optimizer:** AdamW.
- **Schedule:** A two-stage, multi-fidelity approach.
  1.  **Stage 1 (Pre-training):** Generate a much larger (~10,000 cases) but cheaper dataset using RANS simulations. Train the model on this dataset for ~50 epochs to learn the basic flow physics and solution structure.
  2.  **Stage 2 (Fine-tuning):** Load the pre-trained model weights. Fine-tune the model on our high-fidelity WMLES dataset using a lower learning rate. This critical step adapts the model to learn the high-fidelity turbulence and shock interaction physics.
- **Hardware:** Distributed training across multiple GPUs using `jax.pmap`.

---

### Phase 3: Validation & Deployment

This final phase rigorously tests the trained model and prepares it for use.

**Task 3.1: The Triple Crown Gauntlet**

- **Accuracy:**
  1.  Run the trained model on the input conditions of the NASA Falcon 9 SRP validation case.
  2.  Interpolate the predicted surface pressure from the model's output onto the experimental measurement locations.
  3.  Generate the comparative `Cp` plots and compute the final L2 error metric.
- **Speed:**
  1.  Script a single-inference pass of the model on a target GPU.
  2.  Time the end-to-end execution, from data loading to solution output, and report the average over 100 runs.
- **Generalization:**
  1.  Run the model on the entire held-out test set from our generated data.
  2.  Compute the error metrics for each case and plot the error distribution across the parameter space (Mach, altitude, etc.). This will reveal the model's robustness.

**Task 3.2: Model Checkpointing & Release**

- **Final Asset:** A set of trained model weights (e.g., in a `pickle` or `orbax` file) and a simple Python script (`predict.py`) that demonstrates how to:
  1.  Load the model weights.
  2.  Load a new mesh/problem definition.
  3.  Run inference and save the resulting solution field in a standard format (e.g., `.vtu`).
