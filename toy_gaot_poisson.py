"""
Toy GAOT vs Baseline demo on Poisson's equation with a random circular hole
— DOLFINx edition.

This script:

1. Generates synthetic datasets on-the-fly with *DOLFINx* + *Gmsh*
   (arguments: ``--num-samples``, ``--points``).
2. Trains two models in **JAX / Flax**:
     - BaselineMLP  - naive point-wise network
     - ToyGAOT      - simplified geometry-aware operator transformer
3. Prints final relative L2 errors.

Dependencies
------------
conda install -c conda-forge fenics-dolfinx petsc petsc4py gmsh slepc slepc4py
pip install jax flax optax scikit-learn numpy tqdm
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
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem, mesh
import dolfinx.fem.petsc as fem_petsc
import ufl
import gmsh


def _gmsh_square_with_hole(cx: float, cy: float, r: float, mesh_res: int) -> mesh.Mesh:
    """
    Build a unit square [0,1]² with a circular hole using Gmsh and
    convert to a DOLFINx mesh.

    The characteristic length is controlled by ``mesh_res``.
    """
    from dolfinx.io import gmshio

    gmsh.initialize()
    gmsh.model.add("unit_square_with_hole")

    lc = 1.0 / mesh_res  # target element size

    # Square points (counter-clockwise)
    p0 = gmsh.model.geo.addPoint(0.0, 0.0, 0.0, lc)
    p1 = gmsh.model.geo.addPoint(1.0, 0.0, 0.0, lc)
    p2 = gmsh.model.geo.addPoint(1.0, 1.0, 0.0, lc)
    p3 = gmsh.model.geo.addPoint(0.0, 1.0, 0.0, lc)

    l0 = gmsh.model.geo.addLine(p0, p1)
    l1 = gmsh.model.geo.addLine(p1, p2)
    l2 = gmsh.model.geo.addLine(p2, p3)
    l3 = gmsh.model.geo.addLine(p3, p0)
    square = gmsh.model.geo.addCurveLoop([l0, l1, l2, l3])

    # Circle - approximate with 64 straight segments (simpler than arcs)
    n_circle = 64
    circle_pts = []
    circle_lines = []
    prev_pt = None
    for i in range(n_circle):
        theta = 2 * np.pi * i / n_circle
        x = cx + r * np.cos(theta)
        y = cy + r * np.sin(theta)
        pt = gmsh.model.geo.addPoint(float(x), float(y), 0.0, lc)
        circle_pts.append(pt)
        if prev_pt is not None:
            line = gmsh.model.geo.addLine(prev_pt, pt)
            circle_lines.append(line)
        prev_pt = pt
    circle_lines.append(gmsh.model.geo.addLine(circle_pts[-1], circle_pts[0]))
    circle_loop = gmsh.model.geo.addCurveLoop(circle_lines)

    surface = gmsh.model.geo.addPlaneSurface([square, circle_loop])
    gmsh.model.geo.synchronize()
    # Tag the surface so DOLFINx can import it
    pg = gmsh.model.addPhysicalGroup(2, [surface])
    gmsh.model.setPhysicalName(2, pg, "domain")
    gmsh.model.mesh.generate(2)

    msh, *_ = gmshio.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=2)
    gmsh.finalize()
    return msh


def solve_poisson_with_hole(
    cx: float,
    cy: float,
    r: float,
    num_points: int = 256,
    mesh_res: int = 32,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Solve −Δu = 1 on the unit square with a circular hole.

    Dirichlet BCs:
      outer boundary → u = 0
      hole boundary  → u = 1

    Returns
    -------
    coords : (num_points, 2) float32
    u_vals : (num_points, 1) float32
    """
    # ------------------------------------------------------------------ mesh
    msh = _gmsh_square_with_hole(cx, cy, r, mesh_res)

    # ------------------------------------------------------------------ space
    V = fem.functionspace(msh, ("Lagrange", 1))

    # ---------------------------------------------------------------- equation
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    f = fem.Constant(msh, PETSc.ScalarType(1.0))

    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = f * v * ufl.dx

    # ---------------------------------------------------------- Dirichlet BCs
    def _outer(x):
        return (
            np.isclose(x[0], 0.0)
            | np.isclose(x[0], 1.0)
            | np.isclose(x[1], 0.0)
            | np.isclose(x[1], 1.0)
        )

    def _inner(x):
        return (x[0] - cx) ** 2 + (x[1] - cy) ** 2 <= (r + 1e-6) ** 2

    # outer square facets
    facets_out = mesh.locate_entities_boundary(msh, msh.topology.dim - 1, _outer)
    dofs_out = fem.locate_dofs_topological(V, msh.topology.dim - 1, facets_out)
    bc_out = fem.dirichletbc(PETSc.ScalarType(0.0), dofs_out, V)

    # hole facets
    facets_in = mesh.locate_entities_boundary(msh, msh.topology.dim - 1, _inner)
    dofs_in = fem.locate_dofs_topological(V, msh.topology.dim - 1, facets_in)
    bc_in = fem.dirichletbc(PETSc.ScalarType(1.0), dofs_in, V)

    # ---------------------------------------------------------- linear solve
    problem = fem_petsc.LinearProblem(
        a,
        L,
        bcs=[bc_out, bc_in],
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    )
    uh = problem.solve()

    # ---------------------------------------------------------- sampling
    coords = msh.geometry.x.copy().astype(np.float32)
    vals = uh.x.array.real.astype(np.float32)

    if len(coords) > num_points:
        idx = np.random.choice(len(coords), num_points, replace=False)
        coords = coords[idx]
        vals = vals[idx]

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
    """Naive point-wise model: input [x, y, cx, cy, r] → u"""

    @nn.compact
    def __call__(self, coords, params):
        inp = jnp.concatenate([coords, params], axis=-1)
        return MLP((128, 128, 128, 1))(inp)


class SimpleEncoder(nn.Module):
    latent_dim: int
    k: int = 16  # neighbours

    @nn.compact
    def __call__(self, latent_grid_point, coords, params):
        # latent_grid_point: (2,), coords: (B, N, 3), params: (B, N, 3)
        coords_xy = coords[..., :2]  # (B, N, 2)

        dists = jnp.sum((latent_grid_point - coords_xy) ** 2, axis=-1)  # (B, N)
        idx = jnp.argsort(dists, axis=1)[:, : self.k]  # (B, k)

        idx_expanded = jnp.expand_dims(idx, axis=2)
        neighbor_coords = jnp.take_along_axis(coords, idx_expanded, axis=1)
        neighbor_params = jnp.take_along_axis(params, idx_expanded, axis=1)

        mean_coords = jnp.mean(neighbor_coords, axis=1)
        mean_params = jnp.mean(neighbor_params, axis=1)

        latent_grid_b = jnp.broadcast_to(latent_grid_point, (coords.shape[0], 2))

        enc_inp = jnp.concatenate([latent_grid_b, mean_coords[..., :2], mean_params], axis=-1)
        return MLP((128, 128, self.latent_dim))(enc_inp)


class SimpleDecoder(nn.Module):
    k: int = 4  # neighbours

    @nn.compact
    def __call__(self, query, latent_grid, latent_tokens):
        # query: (N, 6), latent_grid: (G, 2), latent_tokens: (G, D)
        query_xy = query[..., :2]  # (N, 2)
        dists = jnp.sum((query_xy[:, None, :] - latent_grid[None, :, :]) ** 2, axis=-1)  # (N, G)
        idx = jnp.argsort(dists, axis=1)[:, : self.k]  # (N, k)

        neighbor_tokens = latent_tokens[idx]
        mean_tokens = jnp.mean(neighbor_tokens, axis=1)

        dec_inp = jnp.concatenate([query, mean_tokens], axis=-1)
        return MLP((128, 128, 1))(dec_inp)


class ToyGAOT(nn.Module):
    latent_dim: int = 64
    grid_size: int = 16

    @nn.compact
    def __call__(self, coords, params):
        # coords: (B, N, 3), params: (B, N, 3)
        grid_1d = jnp.linspace(0.0, 1.0, self.grid_size)
        gx, gy = jnp.meshgrid(grid_1d, grid_1d)
        latent_grid = jnp.stack([gx.ravel(), gy.ravel()], axis=-1) # (G, 2)

        encode = SimpleEncoder(self.latent_dim)
        decode = SimpleDecoder()

        latent_tokens = jax.vmap(encode, in_axes=(0, None, None))(
            latent_grid, coords, params
        ) # (G, B, D)

        latent_tokens_p = jnp.transpose(latent_tokens, (1, 0, 2)) # (B, G, D)

        processed = jax.vmap(MLP((self.latent_dim, self.latent_dim)))(latent_tokens_p) # (B, G, D)

        query_points = jnp.concatenate([coords, params], axis=-1) # (B, N, 6)

        preds = jax.vmap(decode, in_axes=(0, None, 0))(
            query_points, latent_grid, processed
        ) # (B, N, 1)
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
def _train_step(state, coords, params_geom, sols):
    def loss_fn(params):
        preds = state.apply_fn({"params": params}, coords, params_geom)
        return jnp.mean((preds - sols) ** 2)

    val, grads = jax.value_and_grad(loss_fn)(state.params)
    return state.apply_gradients(grads=grads), val


def train_model(
    name, model, data_train, data_test, *, epochs=200, lr=1e-3, batch_size=5
) -> float:
    print(f"\n== Training {name} ==")
    rng = jax.random.PRNGKey(0)
    state = create_state(rng, model, lr, data_train[:2] + ())

    coords_tr, params_tr, sols_tr = data_train
    coords_te, params_te, sols_te = data_test

    n_train = coords_tr.shape[0]

    if n_train == 0:
        print("  (no training samples, skipping optimisation)")
    else:
        loss = jnp.nan
        for ep in range(1, epochs + 1):
            perm = np.random.permutation(n_train)
            for i in range(0, n_train, batch_size):
                idx = perm[i : i + batch_size]
                state, loss = _train_step(
                    state,
                    coords_tr[idx],
                    params_tr[idx],
                    sols_tr[idx],
                )
            if (ep % 50 == 0 or ep == 1) and not jnp.isnan(loss):
                print(f"  epoch {ep:>3d}  loss {loss:.6f}")

    preds = state.apply_fn({"params": state.params}, coords_te, params_te)
    rel_l2 = jnp.linalg.norm(preds - sols_te) / jnp.linalg.norm(sols_te)
    print(f"  -> relative L2 error = {rel_l2:.4f}")
    return float(rel_l2)


# -----------------------------------------------------------------------------#
# 4.  Main                                                                     #
# -----------------------------------------------------------------------------#
def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Toy GAOT demo (DOLFINx)")
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
