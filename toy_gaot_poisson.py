#!/usr/bin/env python3
"""
Toy GAOT vs Baseline demo on Poisson's equation with a random circular hole.

Usage
-----
$ python toy_gaot_poisson.py            # runs end-to-end demo
$ python toy_gaot_poisson.py --help     # see configurable arguments

The script:
1. Generates synthetic datasets on-the-fly with FEniCS (--num-samples, --points).
2. Trains two models in JAX / Flax:
     • BaselineMLP  - naive point-wise network
     • ToyGAOT      - simplified geometry-aware operator transformer
3. Prints final relative L2 errors.

Dependencies
------------
pip install fenics-dolfin jax flax optax scikit-learn numpy tqdm
"""

import argparse
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn
from flax.training import train_state
from tqdm import trange

# -----------------------------------------------------------------------------#
# 1.  Data generator (FEniCS)                                                  #
# -----------------------------------------------------------------------------#
try:
    import dolfin as df  # classic FEniCS
except ModuleNotFoundError as exc:
    print("FEniCS/dolfin not found. Install with:\n  pip install fenics-dolfin")
    raise SystemExit(1) from exc


def solve_poisson_with_hole(
    cx: float,
    cy: float,
    r: float,
    num_points: int = 256,
    mesh_res: int = 32,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Solve -Δu = 1 on square [0,1]² with a circular hole Dirichlet BC.

    Outer boundary: u = 0
    Hole boundary : u = 1

    Returns:
        coords:  (num_points, 2)  float32
        u_vals: (num_points, 1)  float32
    """
    # Build geometry
    domain = df.Rectangle(df.Point(0.0, 0.0), df.Point(1.0, 1.0))
    hole = df.Circle(df.Point(cx, cy), r)
    geom = domain - hole

    mesh = df.generate_mesh(geom, mesh_res)

    # Local refine near hole for accuracy
    cell_markers = df.MeshFunction("bool", mesh, mesh.topology().dim())
    cell_markers.set_all(False)
    for cell in df.cells(mesh):
        p = cell.midpoint()
        if (p.x() - cx) ** 2 + (p.y() - cy) ** 2 < (2 * r) ** 2:
            cell_markers[cell] = True
    mesh = df.refine(mesh, cell_markers)

    V = df.FunctionSpace(mesh, "P", 1)

    # Boundary conditions
    u_outer = df.Constant(0.0)

    def outer_boundary(x, on_bnd):
        return on_bnd and (
            df.near(x[0], 0) or df.near(x[0], 1) or df.near(x[1], 0) or df.near(x[1], 1)
        )

    bc_outer = df.DirichletBC(V, u_outer, outer_boundary)

    u_inner = df.Constant(1.0)

    def inner_boundary(x, on_bnd):
        return on_bnd and ((x[0] - cx) ** 2 + (x[1] - cy) ** 2 <= (r + 1e-3) ** 2)

    bc_inner = df.DirichletBC(V, u_inner, inner_boundary)

    bcs = [bc_outer, bc_inner]

    # Variational problem
    u = df.TrialFunction(V)
    v = df.TestFunction(V)
    f = df.Constant(1.0)
    a = df.dot(df.grad(u), df.grad(v)) * df.dx
    L = f * v * df.dx

    u_sol = df.Function(V)
    df.solve(a == L, u_sol, bcs)

    coords = mesh.coordinates().astype(np.float32)
    vals = u_sol.compute_vertex_values(mesh).astype(np.float32)

    # Sub-sample for fixed tensor size
    if len(coords) > num_points:
        idx = np.random.choice(len(coords), num_points, replace=False)
        coords, vals = coords[idx], vals[idx]

    coords = jnp.asarray(coords)
    vals = jnp.asarray(vals)[:, None]
    return coords, vals


def generate_dataset(
    rng: np.random.Generator, n_samples: int, n_points: int
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    coords_all, params_all, sols_all = [], [], []
    for _ in trange(n_samples, desc="Generating"):
        cx = rng.uniform(0.3, 0.7)
        cy = rng.uniform(0.3, 0.7)
        r = rng.uniform(0.1, 0.2)

        coords, sol = solve_poisson_with_hole(cx, cy, r, n_points)
        params = jnp.tile(
            jnp.asarray([cx, cy, r], dtype=jnp.float32), (coords.shape[0], 1)
        )

        coords_all.append(coords)
        params_all.append(params)
        sols_all.append(sol)
    return jnp.stack(coords_all), jnp.stack(params_all), jnp.stack(sols_all)


# -----------------------------------------------------------------------------#
# 2.  Models                                                                   #
# -----------------------------------------------------------------------------#
class MLP(nn.Module):
    features: Tuple[int, ...]

    @nn.compact
    def __call__(self, x):
        for feat in self.features[:-1]:
            x = nn.Dense(feat)(x)
            x = nn.relu(x)
        x = nn.Dense(self.features[-1])(x)
        return x


class BaselineMLP(nn.Module):
    """Naive point-wise model: input [x,y,cx,cy,r] → u"""

    @nn.compact
    def __call__(self, coords, params):
        inp = jnp.concatenate([coords, params], axis=-1)
        return MLP((128, 128, 128, 1))(inp)


class SimpleEncoder(nn.Module):
    latent_dim: int
    k: int = 16  # neighbours

    @nn.compact
    def __call__(self, latent_grid, coords, params):
        from sklearn.neighbors import NearestNeighbors

        nbrs = NearestNeighbors(n_neighbors=self.k).fit(np.asarray(coords))
        _, idx = nbrs.kneighbors(np.asarray(latent_grid))
        neighbor_coords = coords[idx]
        neighbor_params = params[idx]

        mean_coords = jnp.mean(neighbor_coords, axis=1)
        mean_params = jnp.mean(neighbor_params, axis=1)
        enc_inp = jnp.concatenate([latent_grid, mean_coords, mean_params], axis=-1)
        return MLP((128, 128, self.latent_dim))(enc_inp)


class SimpleDecoder(nn.Module):
    k: int = 4  # neighbours

    @nn.compact
    def __call__(self, query, latent_grid, latent_tokens):
        from sklearn.neighbors import NearestNeighbors

        nbrs = NearestNeighbors(n_neighbors=self.k).fit(np.asarray(latent_grid))
        _, idx = nbrs.kneighbors(np.asarray(query))
        mean_tokens = jnp.mean(latent_tokens[idx], axis=1)
        dec_inp = jnp.concatenate([query, mean_tokens], axis=-1)
        return MLP((128, 128, 1))(dec_inp)


class ToyGAOT(nn.Module):
    latent_dim: int = 64
    grid_size: int = 16

    @nn.compact
    def __call__(self, coords, params):
        grid_1d = jnp.linspace(0.0, 1.0, self.grid_size)
        gx, gy = jnp.meshgrid(grid_1d, grid_1d)
        latent_grid = jnp.stack([gx.ravel(), gy.ravel()], axis=-1)

        encode = SimpleEncoder(self.latent_dim)
        decode = SimpleDecoder()

        latent_tokens = jax.vmap(encode, in_axes=(None, 0, 0))(
            latent_grid, coords, params
        )
        processed = MLP((self.latent_dim, self.latent_dim))(latent_tokens)
        preds = jax.vmap(decode, in_axes=(0, None, 0))(coords, latent_grid, processed)
        return preds


# -----------------------------------------------------------------------------#
# 3.  Training utilities                                                       #
# -----------------------------------------------------------------------------#
def create_state(rng_key, model, lr, batch):
    params = model.init(rng_key, *batch)["params"]
    return train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=optax.adam(lr)
    )


@jax.jit
def loss_fn(params, apply_fn, coords, params_geom, sols):
    preds = apply_fn({"params": params}, coords, params_geom)
    return jnp.mean((preds - sols) ** 2)


@jax.jit
def train_step(state, coords, params_geom, sols):
    val, grads = jax.value_and_grad(loss_fn)(
        state.params, state.apply_fn, coords, params_geom, sols
    )
    return state.apply_gradients(grads=grads), val


def train_model(name, model, data_train, data_test, epochs=200, lr=1e-3, batch_size=5):
    print(f"\n== Training {name} ==")
    rng = jax.random.PRNGKey(0)
    state = create_state(rng, model, lr, data_train[:2] + ())

    coords_tr, params_tr, sols_tr = data_train
    coords_te, params_te, sols_te = data_test

    n_train = coords_tr.shape[0]
    for ep in range(1, epochs + 1):
        perm = np.random.permutation(n_train)
        for i in range(0, n_train, batch_size):
            idx = perm[i : i + batch_size]
            state, loss = train_step(
                state,
                coords_tr[idx],
                params_tr[idx],
                sols_tr[idx],
            )
        if ep % 50 == 0 or ep == 1:
            print(f"  epoch {ep:>3d}  loss {loss:.6f}")

    # evaluation
    preds = state.apply_fn({"params": state.params}, coords_te, params_te)
    rel_l2 = jnp.linalg.norm(preds - sols_te) / jnp.linalg.norm(sols_te)
    print(f"  -> relative L2 error = {rel_l2:.4f}")
    return float(rel_l2)


# -----------------------------------------------------------------------------#
# 4.  Main                                                                     #
# -----------------------------------------------------------------------------#
def main(argv=None):
    parser = argparse.ArgumentParser(description="Toy GAOT demo")
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--points", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=200)
    args = parser.parse_args(argv)

    rng_np = np.random.default_rng(0)
    coords, params_geom, sols = generate_dataset(rng_np, args.num_samples, args.points)

    split = int(0.8 * args.num_samples)
    data_train = (coords[:split], params_geom[:split], sols[:split])
    data_test = (coords[split:], params_geom[split:], sols[split:])

    baseline_err = train_model(
        "BaselineMLP", BaselineMLP(), data_train, data_test, epochs=args.epochs
    )
    gaot_err = train_model(
        "ToyGAOT", ToyGAOT(), data_train, data_test, epochs=args.epochs
    )

    print("\n== Comparison ==")
    print(f"BaselineMLP error : {baseline_err:.4f}")
    print(f"ToyGAOT   error   : {gaot_err:.4f}")


if __name__ == "__main__":
    main()
