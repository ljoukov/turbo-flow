# GOALS.md — High-Level Objectives for Project Turbo-Flow

1. Build a neural-operator surrogate that predicts steady-state supersonic retro-propulsion (SRP) flow fields with:
   • Accuracy < 5 % L2 error on surface pressure for the NASA Falcon-9 SRP case  
   • Speed < 10 minutes per full 3-D field on a single A100/H100 GPU  
   • Generalization < 10 % error on unseen Mach/altitude/throttle/AoA combinations

2. Replace hours-to-days high-fidelity CFD runs with near-interactive predictions, enabling rapid design iteration for planetary entry and landing vehicles.

3. Produce an open, reproducible workflow:
   • Public code, training data (where licenses allow), and pretrained weights  
   • One-command inference script for new geometries / conditions

4. Lay the foundation for future extensions:
   • Time-dependent SRP  
   • Transfer to different vehicle geometries via zero-shot generalization  
   • Integration into closed-loop flight-control simulations
