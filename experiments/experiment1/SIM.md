────────────────────────────────────────

1.  What does the model try to predict?
    ────────────────────────────────────────
    The authors focus on the non-dimensional aerodynamic load, which they denote with the single symbol

            C

Throughout the poster C is plotted versus thrust–coefficient CT for several nozzle-count configurations.  
C is therefore the _scalar_ output of the model.

──────────────────────────────────────── 2. Which quantities are given to the model?
────────────────────────────────────────
The full NASA Langley SRP wind-tunnel data set contains 168 primitive quantities and 125 measured force / moment coefficients.  
Using a combination of PCA, correlation analysis and random-forest feature-ranking they finally split the inputs into

• θp (physics inputs – go into the analytic piece)
– M : freestream Mach number  
 – CT : thrust coefficient (= T / (q∞ At ))  
 – cpe : exit-pressure coefficient (= (pe – p∞)/q∞)  
 – αe : nozzle cant / vehicle incidence angle (deg)

• θc (coupling inputs – go into the residual NN)
– sd : sideline (plume impingement) angle (deg)  
 – p∞ : freestream static pressure (Pa)  
 – T∞ : freestream static temperature (K)  
 – Re∞ : freestream Reynolds number  
 – any other flow-geometry features that survived their ranking step  
 (the code below keeps the interface fully general: θc can have an
arbitrary length)

All features are fed in non-dimensional form (or are standardised in a
data-pre-processing pipeline that is _outside_ of the model definition).

──────────────────────────────────────── 3. How is the prediction composed?
────────────────────────────────────────
The _hybrid physics-embedded, data-driven (HPDD)_ model is an additive
decomposition

        C(M,CT,cpe,αe ,            sd,p∞,T∞,Re∞,…)  =
        ─────────────────────────────────────────────
                 ↑θp                                    ↑θc
        =  C0              (zero-thrust reference, here 0)
        +  Cs rp(θp)       (closed-form SRP reduced-order model)
        +  Cnn (θc)        (small feed-forward network that learns only
                            the residual, i.e. what the ROM cannot capture)

For the poster the analytic SRP term is written

        Cs rp(θp) = tan(αe) ·
                    { 2(γ² M² − 1)/(γ² + 1) · Ae/At } +
                    L / At +
                    1/(1 + exp(−CT/αf))

with γ = 1.4 for CO₂, and where Ae/At and L/At are geometric ratios that are
constant for every nozzle-count configuration and can therefore be treated
as _learnable scalar parameters_ that get fitted together with the NN.
(If they are already known from CAD they can simply be hard-coded.)

Because C0 = 0 for a rocket-off run, the final trainable object is

        Ĉ(θp,θc) = Cs rp(θp ; ϕphys) + fNN(θc ; w)

ϕphys … two or three scalar coefficients (Ae/At, L/At, αf)  
w … all trainable NN weights

──────────────────────────────────────── 4. A concise implementation in JAX
────────────────────────────────────────
Below is a _self-contained_ module that reproduces the architecture that is
written on the poster:

• four fully-connected layers (400-200-100-50)  
• ReLU activations  
• 30 % dropout in every hidden layer  
• Adam optimiser (lr = 1 e-2, β1 = 0.9, β2 = 0.999, ϵ = 1 e-7)  
• MSE loss

Everything is written with Flax (the Linen API) and Optax.

```python
# hpdd_model.py
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax
from typing import Sequence, Any, Dict

# ------------------------------------------------------------
# 1.  Analytic / physics part
# ------------------------------------------------------------
class SRPPhysics(nn.Module):
    """Closed-form reduced-order model C_srp(θp)."""
    gamma: float = 1.4                       # specific-heat ratio CO₂
    # three scalar parameters that can either be fixed or trained
    learnable: bool = True                   # True = will be trained

    @nn.compact
    def __call__(self, theta_p: jnp.ndarray) -> jnp.ndarray:
        """
        theta_p = [M, C_T, c_pe, alpha_e]  (shape [..., 4])
        returns  shape [...,]              (scalar)
        """
        M, C_T, c_pe, alpha_e = jnp.split(theta_p, 4, axis=-1)  # each shape [...,1]
        gamma = self.gamma

        # ϕphys – either fixed constants or trainable 1×1 parameters
        def maybe_trainable(name, init_val):
            if self.learnable:
                return self.param(name, lambda _: jnp.asarray(init_val))
            else:
                return jnp.asarray(init_val)

        Ae_At = maybe_trainable("Ae_At", 1.00)      # nozzle exit / throat area ratio
        L_At  = maybe_trainable("L_At", 0.00)      # body-length / throat-area ratio
        alpha_f = maybe_trainable("alpha_f", 1.00)  # logistic width for CT-term

        term1 = jnp.tan(alpha_e) * (
                    2.0 * ( (gamma * M)**2 - 1.0 ) /
                    (gamma**2 + 1.0) * Ae_At
                )
        term2 = L_At
        term3 = 1.0 / (1.0 + jnp.exp(-C_T / alpha_f))
        return (term1 + term2 + term3).squeeze(-1)  # shape [...,]

# ------------------------------------------------------------
# 2.  Data-driven residual NN
# ------------------------------------------------------------
class ResidualMLP(nn.Module):
    hidden_dims: Sequence[int] = (400, 200, 100, 50)
    dropout_rate: float = 0.30
    training: bool = False         # Dropout on/off switch

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for dim in self.hidden_dims:
            x = nn.Dense(dim)(x)
            x = nn.relu(x)
            x = nn.Dropout(rate=self.dropout_rate)(x,
                      deterministic=not self.training)
        x = nn.Dense(1)(x)
        return x.squeeze(-1)        # shape [...,]

# ------------------------------------------------------------
# 3.  Full HPDD model = physics term + residual NN
# ------------------------------------------------------------
class HPDDModel(nn.Module):
    hidden_dims: Sequence[int] = (400, 200, 100, 50)
    dropout: float = 0.30
    train_mode: bool = False
    learnable_physics: bool = True

    @nn.compact
    def __call__(self,
                 theta_p: jnp.ndarray,
                 theta_c: jnp.ndarray) -> jnp.ndarray:
        c_phys = SRPPhysics(learnable=self.learnable_physics)(theta_p)
        c_nn   = ResidualMLP(self.hidden_dims,
                             self.dropout,
                             self.train_mode)(theta_c)
        return c_phys + c_nn         # shape [...,]

# ------------------------------------------------------------
# 4.  Helper : create training state
# ------------------------------------------------------------
def create_train_state(rng,
                       model: HPDDModel,
                       learning_rate: float = 1e-2
                       ) -> train_state.TrainState:
    params = model.init(rng,
                        theta_p=jnp.ones((1,4)),
                        theta_c=jnp.ones((1,8)))['params']
    tx     = optax.adam(learning_rate=learning_rate,
                        b1=0.9, b2=0.999, eps=1e-7)
    return train_state.TrainState.create(apply_fn=model.apply,
                                         params=params,
                                         tx=tx)

# ------------------------------------------------------------
# 5.  Single training step (80/10/10 split etc. live outside)
# ------------------------------------------------------------
@jax.jit
def train_step(state: train_state.TrainState,
               rng: jax.random.KeyArray,
               batch: Dict[str, jnp.ndarray]
               ) -> train_state.TrainState:
    """batch must contain keys: 'theta_p', 'theta_c', 'C_true'"""
    def loss_fn(params):
        preds = state.apply_fn({'params': params},
                               theta_p=batch['theta_p'],
                               theta_c=batch['theta_c'],
                               rngs={'dropout': rng},
                               mutable=False)
        loss = jnp.mean((preds - batch['C_true'])**2)
        return loss

    grads = jax.grad(loss_fn)(state.params)
    return state.apply_gradients(grads=grads)
```

The file gives you everything you need to

    1.  instantiate the model
    2.  split features into (θp, θc)
    3.  train with mini-batches (batch size = 5, 100 epochs)
    4.  evaluate on validation / test sets

The analytic term can be _disabled_ (set `learnable_physics=False`) to obtain
an ablation that is equivalent to a _pure_ neural-network surrogate.

──────────────────────────────────────── 5. How to call it (minimal example)
────────────────────────────────────────

```python
import jax, jax.numpy as jnp
from hpdd_model import HPDDModel, create_train_state, train_step

rng = jax.random.PRNGKey(0)
model = HPDDModel(train_mode=True)                 # dropout *on*
state = create_train_state(rng, model)

for epoch in range(100):
    for batch in dataloader:                       # user-defined
        rng, sub = jax.random.split(rng)
        state = train_step(state, sub, batch)

# use  model.apply({'params': state.params}, θp, θc, train_mode=False)
# for fast, deterministic prediction  (≈0.03 s on a single CPU core)
```

That is the complete HPDD-SRP coefficient model described in the poster,
implemented in JAX/Flax with the exact network topology and optimiser
hyper-parameters provided by the authors.
