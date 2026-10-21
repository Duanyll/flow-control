# Sampling math

What each sampler node computes, in the notation of [notation.md](notation.md). Config fields, frames and the layer structure are in [../sampler/nodes.md](../sampler/nodes.md). Code paths are under `flow_control/samplers/` unless stated.

Inside one step, $x$ is the row's latents at $\sigma_i$, $\hat{v}$ the velocity of the step's prediction at $(x, \sigma_i)$, and $\hat{x}_0$, $\hat{\epsilon}$ follow from them.

## Forward process

### `renoise`

`trajectory.py` `Renoise` moves the row from σ to $\sigma_e$ = `sigma_end` without a model call:

$$
x \leftarrow c\,x + \sqrt{\sigma_e^2 - c^2\sigma^2}\;\xi, \qquad c = \frac{1-\sigma_e}{1-\sigma}
$$

Applied to $x_\sigma = (1-\sigma)\,x_0 + \sigma\epsilon$, the result is $(1-\sigma_e)\,x_0$ plus Gaussian noise of std $\sigma_e$. $\sigma_e = \sigma$ is a no-op; $\sigma_e < \sigma$ raises (negative square root, or division by zero at $\sigma = 1$).

### `init_latent`

`InitLatent` sets $x = x_{\text{src}}$ (fp32) and $\sigma = 0$, then runs `renoise` to $\sigma_e$:

$$
x = (1-\sigma_e)\,x_{\text{src}} + \sigma_e\,\xi
$$

The SDEdit start (Meng et al., 2022). The default $\sigma_e = 1$ keeps nothing of the image.

## Grids

### Grid points

`BaseGrid.sigmas` (`trajectory.py`), with $\sigma_s$ the row's σ when the grid starts, $\sigma_e$ = `sigma_end` and $F$ = `shift.curve(row, n_full)`:

$$
n_{\text{full}} = \mathrm{round}\left(\frac{n}{|\sigma_s - \sigma_e|}\right)
$$

$$
u_k = F^{-1}(\sigma_s) + \frac{k}{n}\left(F^{-1}(\sigma_e) - F^{-1}(\sigma_s)\right), \qquad k = 0, \dots, n
$$

$$
\sigma_0 = \sigma_s, \qquad \sigma_k = F(u)_k \quad (0 < k < n), \qquad \sigma_n = \sigma_e
$$

$F$ maps the whole $u$ vector at once, because the terminal stretch reads its last points; the endpoints are then set exactly. Since $u$ is linear between the inverses of the endpoints, a grid that starts or ends mid-way puts its points on the curve of the full 1 → 0 grid. $n_{\text{full}}$ measures the span in σ, not in $u$: under a step-dependent shift (`flux2`) the pieces of a split grid get other factors than the whole grid.

`custom_sigmas` replaces this: $\sigma_k$ is the list, which must hold $n + 1$ values, start within $10^{-2}$ of the row's σ and end within $10^{-6}$ of $\sigma_e$. The row's σ is set to $\sigma_0$ and $x$ is kept, so the initial noise is evaluated as if at $\sigma_0$ (official grids start just below 1).

`Grid.run` takes transition $i$ from $\sigma_i$ to $\sigma_{i+1}$ for $i = 0, \dots, n-1$. A grid with $\sigma_e > \sigma_s$ runs upward (inversion).

### Shift curves

Every member of `shift_registry` but `logit_normal` is a `ShiftCurve` (`shift.py`) with the factor $s$ of `BaseShift.factor(L, n_full)`:

$$
f(u) = \frac{s\,u}{1 + (s-1)\,u}, \qquad f^{-1}(\sigma) = \frac{\sigma}{\sigma + s\,(1-\sigma)}
$$

$s = 1$ is the identity; diffusers' exponential time shift with parameter `mu` is $s = e^{\texttt{mu}}$.

The sequence length (`BaseShift._seq_len`), with `image_size` $= (H, W)$ and `model_image_size` $= (H_m, W_m)$ (default `image_size`; a tile's size under tiling), is $L = \lfloor N H_m W_m / (H W) \rfloor$ for `image_seq_len_from: "actual"`, with $N$ the tokens of `noisy_latents`, and $L = \lfloor H_m W_m / 256 \rfloor$ for `"image_size"`.

**`constant`** (`ConstantShift`): $s$ = `shift_value`.

**`linear`** (`LinearShift`), FLUX.1's resolution-dependent shift (diffusers `calculate_shift`), with $(L_0, k_0)$ = (`base_image_seq_len`, `base_shift`) and $(L_1, k_1)$ = (`max_image_seq_len`, `max_shift`):

$$
\ln s = k_0 + (L - L_0)\,\frac{k_1 - k_0}{L_1 - L_0}
$$

**`squared`** (`SquaredShift`): $s = \sqrt{L / L_0}$, with $L_0$ = `base_image_seq_len`.

**`flux2`** (`Flux2Shift`), FLUX.2's empirical `mu` (diffusers `compute_empirical_mu`), with $m_{10} = a_1 L + b_1$ and $m_{200} = a_2 L + b_2$ from the fields `a1`, `b1`, `a2`, `b2`, and the fields $c = 190$, $d = 200$, $L_t$ = `image_seq_len_threshold`. Below the threshold $\ln s$ is linear in the step count, $m_{10}$ at 10 steps and $m_{200}$ at 200:

$$
\ln s = \begin{cases} m_{200} & L > L_t \\ m_{200} + (m_{200} - m_{10})\,\dfrac{n_{\text{full}} - d}{c} & \text{otherwise} \end{cases}
$$

**`logit_normal`** (`LogitNormalShift`, `LogitNormalCurve`), Ideogram 4's schedule, evaluated in float64, with $a$ = `mean` $+ \frac{1}{2}\ln(H_m W_m / P)$, $P$ = `base_pixels`, $b$ = `std` and $\Phi$ the standard normal CDF. It ignores $L$, $n_{\text{full}}$, `shift_terminal` and `diffusers_grid`:

$$
F(u) = \mathrm{expit}\left(a + b\,\Phi^{-1}(u)\right), \qquad F^{-1}(\sigma) = \Phi\left(\frac{\mathrm{logit}(\sigma) - a}{b}\right)
$$

### Terminal stretch

`ShiftCurve.__call__` applies `shift_terminal` $= T$ when it is set and non-zero, $n \ge 2$, and the grid ends at $u_n = 0$ ($\sigma_e = 0$). With $\sigma_k = f(u_k)$:

$$
\sigma_k \leftarrow 1 - (1-\sigma_k)\,\frac{1-T}{1-\sigma_{n-1}} \qquad (\sigma_k > 0)
$$

So $\sigma_{n-1}$ becomes $T$ and $\sigma = 1$ stays 1 (diffusers `stretch_shift_to_terminal`). The stretch uses the grid's own last positive σ; $F^{-1} = f^{-1}$ ignores it.

### `diffusers_grid`

`DiffusersCurve`, with $N$ = `diffusers_grid`, on full 1 → 0 grids only (raises otherwise), ignoring `shift_terminal`:

$$
\sigma_k = f\left(1 - (1-a)\,\frac{k}{n-1}\right) \quad (k < n), \qquad \sigma_n = 0, \qquad a = f(1/N)
$$

This is diffusers' `FlowMatchEulerDiscreteScheduler.set_timesteps` with a static shift, which shifts its lowest training σ at construction and the inference linspace again.

### `sde_window_grid`

`SdeWindowGrid` with $(a, b)$ = `window_range`, default $(0, n-1)$. `select_sde_window` draws the start $j$ uniformly from $\lbrace a, \dots, b - \texttt{size} \rbrace$ with the row's generator, once per run. Steps $j \le i < j + \texttt{size}$ use the transition with η = the grid's `eta` (`SdeWindowGrid._sde`); the others use the transition as configured. The default range excludes the last step. It raises unless $0 \le a < b < n$ and $0 < \texttt{size} \le b - a$.

## Transitions

`GaussianTransition.run` (`transition.py`): a step is stochastic when `is_stochastic` holds, that is $\eta \ne 0$ and $0 < \sigma_{i+1} < \sigma_i$. Then

$$
x_{\sigma_{i+1}} = \mu_i + \delta_i\,\xi
$$

with $(\mu_i, \delta_i, g_i)$ from `moments`. Otherwise the step is `deterministic`, by default the Euler ODE step:

$$
x_{\sigma_{i+1}} = x + \Delta\sigma_i\,\hat{v}
$$

The step to σ = 0 and upward steps are deterministic for every member.

### `euler`

`Euler`, `euler_moments`:

$$
g_i = \eta\sqrt{\frac{\sigma_i}{1 - \sigma_i}} \quad (\sigma_i < 1), \qquad g_i = \eta\sqrt{\frac{1}{1 - \sigma_{i+1}}} \quad (\sigma_i = 1)
$$

$$
\mu_i = x\left(1 + \frac{g_i^2\,\Delta\sigma_i}{2\sigma_i}\right) + \hat{v}\left(1 + \frac{g_i^2\,(1-\sigma_i)}{2\sigma_i}\right)\Delta\sigma_i = x + \left(\hat{v} + \frac{g_i^2}{2\sigma_i}\,\hat{\epsilon}\right)\Delta\sigma_i
$$

$$
\delta_i = g_i\sqrt{-\Delta\sigma_i}
$$

Flow-GRPO's SDE (Liu et al., 2025): an Euler–Maruyama step of the reverse SDE with diffusion $g$, whose drift subtracts $\frac{1}{2}g^2$ times the score estimate $-\hat{\epsilon}/\sigma_i$ so the marginals match the ODE's. $\sigma/(1-\sigma)$ is infinite at σ = 1, so the first step of a grid from 1 takes $\sigma_{i+1}$ in the denominator.

### `ddim`

`Ddim`, `ddim_moments`:

$$
\hat{\epsilon} = \frac{x - (1-\sigma_i)\,\hat{x}_0}{\sigma_i}
$$

$$
\delta_i = \eta\,\sigma_{i+1}\sqrt{\max\left(1 - \frac{\sigma_{i+1}^2\,(1-\sigma_i)^2}{\sigma_i^2\,(1-\sigma_{i+1})^2},\ 0\right)}, \qquad g_i = \delta_i
$$

$$
\mu_i = (1-\sigma_{i+1})\,\hat{x}_0 + \sqrt{\max\left(\sigma_{i+1}^2 - \delta_i^2,\ 0\right)}\;\hat{\epsilon}
$$

DDIM (Song et al., 2021) with signal coefficient $1 - \sigma$ and noise coefficient σ; on a downward step $\delta_i = \eta\,\sigma_{i+1}\sqrt{1 - e^{-2h_i}}$. The deterministic step is $\mu_i$ at η = 0, $(1-\sigma_{i+1})\,\hat{x}_0 + \sigma_{i+1}\hat{\epsilon}$, equal to the Euler step in exact arithmetic. It divides by $\sigma_i$, so it is NaN at $\sigma_i = 0$ (an inversion grid starting at 0).

### `cps`

`Cps.moments`, with $\hat{\epsilon} = x + (1-\sigma_i)\,\hat{v}$:

$$
\delta_i = g_i = \sigma_{i+1}\sin\frac{\eta\pi}{2}, \qquad \mu_i = (1-\sigma_{i+1})\,\hat{x}_0 + \sqrt{\sigma_{i+1}^2 - \delta_i^2}\;\hat{\epsilon} = (1-\sigma_{i+1})\,\hat{x}_0 + \sigma_{i+1}\cos\frac{\eta\pi}{2}\;\hat{\epsilon}
$$

Coefficients-Preserving Sampling (Wang & Yu, 2025): fresh and predicted noise together have std $\sigma_{i+1}$, so $x_{\sigma_{i+1}}$ keeps the interpolation's noise coefficient at every η. Its density is an unnormalized residual ([log-density](#log-density)).

### `dance`

`Dance.moments`, with the code's `score_estimate` $-(x - (1-\sigma_i)\,\hat{x}_0)/\sigma_i^2 = -\hat{\epsilon}/\sigma_i$:

$$
\mu_i = x + \Delta\sigma_i\,\hat{v} - \frac{\eta^2}{2}\left(-\frac{\hat{\epsilon}}{\sigma_i}\right)\Delta\sigma_i = x + \left(\hat{v} + \frac{\eta^2}{2\sigma_i}\,\hat{\epsilon}\right)\Delta\sigma_i, \qquad \delta_i = g_i = \eta\sqrt{\sigma_i - \sigma_{i+1}}
$$

DanceGRPO's SDE (Xue et al., 2025): the `euler` SDE with the constant diffusion $g = \eta$.

### `flash`

`Flash.moments`, at index $i$ of the enclosing grid's $n_g$ steps (`caller(row, BaseGrid)`), with $\ell_s, \ell_e$ = `noise_scale_start`, `noise_scale_end`:

$$
\ell_i = \ell_s + (\ell_e - \ell_s)\,\frac{i}{n_g - 1} \quad (\ell_i = \ell_s \text{ if } n_g = 1), \qquad \mu_i = (1-\sigma_{i+1})\,\hat{x}_0, \qquad \delta_i = g_i = \sigma_{i+1}\,\ell_i
$$

`Flash.noise` clips the draw elementwise to $\pm\chi\,\mathrm{std}(\xi)$, with $\chi$ = `noise_clip_std` $> 0$ and the Bessel-corrected std over all elements of the draw. η only makes the step stochastic (default 1); its value does not enter. `log_prob` uses the unclipped Gaussian. HiDream-O1 Dev's sampler: $\hat{x}_0$ re-noised with fresh noise at a reduced scale.

### Log-density

`GaussianTransition.log_prob` with `normal_log_prob`, for a stochastic step and the recorded `next_latents` $x_{\sigma_{i+1}}$ (detached):

$$
\log p(x_{\sigma_{i+1}} \mid x_{\sigma_i}) = \frac{1}{ND}\sum_{k=1}^{ND}\left[-\frac{\left(x_{\sigma_{i+1}} - \mu_i\right)_k^2}{2\delta_i^2} - \log\delta_i - \frac{1}{2}\log 2\pi\right]
$$

It is a mean over the sample's elements, not a sum, and its gradient flows only through $\mu_i$, that is through $\hat{v}$. `Cps.density` replaces it with the residual

$$
\log p = -\frac{1}{ND}\sum_{k=1}^{ND}\left(x_{\sigma_{i+1}} - \mu_i\right)_k^2
$$

A deterministic step returns $\log p = 0$ with std 0. `StepLogProb` carries $(\log p, \mu_i, g_i)$. GRPO records $\log p$ at rollout time and recomputes it in replay under `train_predictor` ([objectives.md](objectives.md)).

### `differential_diffusion`

`DifferentialDiffusion.run`, before its inner transition, at index $i$ of the enclosing grid's $n_g$ steps:

$$
x_{\text{ref}} = (1-\sigma_i)\,x_{\text{src}} + \sigma_i\,\epsilon, \qquad x \leftarrow \mathbf{1}[1 - q > i/n_g] \odot x_{\text{ref}} + \mathbf{1}[1 - q \le i/n_g] \odot x
$$

$q$ is `inpaint_mask_latents`, one strength per patch pixel, repeated over the channels of its token. $x_{\text{src}}$ is the `from` latents (default `inpaint`). $\epsilon$ is the row's latents at the node's first step (`_noise`), the initial noise because the grid starts at σ = 1. An element with strength $q$ follows the noised reference until step $\lceil (1-q)\,n_g \rceil$: $q = 1$ is free from the first step, $q = 0$ is never released. Differential Diffusion (Levin & Fried, 2023).

## Predictions

### `model`

`Model` (`prediction.py`):

$$
\hat{v} = v_\theta(x, \sigma, c)
$$

θ is `Model.variant`, else the row's `RootFrame.variant`.

### `cfg`

`Cfg.run`, `Cfg.guide`, with γ = `scale`. At γ = 1, $\hat{v} = v_c$ from one call. Otherwise both branches run in one round: $v_c$ is `prediction` at $(x, \sigma, c)$, and $\bar{v}$ is `negative` (default: `prediction`) at $(x, \sigma, \bar{c})$ for `negative_condition: "negative"`, or at $(x, \sigma, c)$ for `"positive"` (autoguidance by a `negative` tree under another variant).

$$
v_\gamma = \bar{v} + \gamma\,(v_c - \bar{v})
$$

$\hat{v} = v_\gamma$, or with `renorm`, per token:

$$
\hat{v} = v_\gamma \cdot \mathrm{clamp}\left(\frac{\lVert v_c\rVert_D}{\lVert v_\gamma\rVert_D + \epsilon_r},\ a,\ b\right)
$$

with $\epsilon_r$ = `renorm_eps`, $a$ = `renorm_min`, $b$ = `renorm_max` (null: no upper bound). The default $b = 1$ only shrinks tokens whose guided norm exceeds the conditional one; $b$ = null rescales every token to $\lVert v_c\rVert_D$ (Qwen-Image's true CFG).

### `cfg_pp`

`CfgPlusPlus.guide`, `kappa`. Both branches always run, γ = 1 included:

$$
\hat{v} = \bar{v} + \kappa\,(v_\gamma - \bar{v})
$$

$v_\gamma$ is `cfg`'s output, renorm included. κ reads the enclosing `euler` or `ddim` step's $\sigma_i$, $\sigma_{i+1}$ and η; $\kappa = 1$ at $\sigma_i = 1$, otherwise

$$
\kappa_{\text{euler}} = \frac{\sigma_i\,(1-\sigma_{i+1})}{(1 + \eta^2/2)\,(\sigma_i - \sigma_{i+1})}, \qquad \kappa_{\text{ddim}} = \frac{\sigma_i\,(1-\sigma_{i+1})}{\sigma_i\,(1-\sigma_{i+1}) - (1-\sigma_i)\sqrt{\max\left(\sigma_{i+1}^2 - \delta_i^2,\ 0\right)}}
$$

with $\delta_i$ the `ddim` noise std. κ makes the step's mean, written in $\hat{x}_0$ and $\hat{\epsilon}$, take $\hat{x}_0$ from $v_\gamma$ and $\hat{\epsilon}$ from $\bar{v}$: CFG++ (Chung et al., 2024) for the flow step. κ uses the node's η also on steps the node takes deterministically: on an `euler` step to σ = 0 with η ≠ 0 it is $1/(1 + \eta^2/2)$, where the deterministic step needs 1.

### `tiled`

`Tiled.run` with `flow_control/utils/tiling.py`. `plan_tiles` lays tiles on the token grid per axis, with extent $E$, tile $T$ and overlap $O$ (the `TileLayout` lengths divided by `stride`): $E \le T$ gives one tile $[0, E)$; otherwise $k = \lceil (E - O)/(T - O) \rceil$ tiles of length $T$ start at $\mathrm{round}(j\,(E - T)/(k - 1))$, $j = 0, \dots, k-1$. The 2-D plan is the product of the two axes. Tile $j$ predicts $v_j$ on its crop of $x$, with its own condition (`row["tiles"][j]`, else the row's). `stitch_tiles` blends in fp32 over the tiles covering each token:

$$
\hat{v} = \frac{\sum_j \phi_j \odot v_j}{\sum_j \phi_j}
$$

$\phi_j$ is the product of a 1-D window per axis. Over a tile of $S$ tokens, at token $y = 0, \dots, S-1$:

$$
\phi(y) = \sin^2\left(\frac{\pi}{2}\min\left(\frac{y + 1/2}{R_-},\ 1\right)\right)\sin^2\left(\frac{\pi}{2}\min\left(\frac{S - y - 1/2}{R_+},\ 1\right)\right)
$$

$R_-$ and $R_+$ are the deepest penetration of an overlapping tile from the low and the high side (`_neighbor_ramps`); a factor is 1 when its $R$ is 0. Across a planned overlap the two neighbours' ramps are $\sin^2$ and $\cos^2$ of the same angle, so the weights sum to 1.

### `momentum`

`flow_control/contrib/momentum_guidance.py` `Momentum`, with $v$ the child's velocity and $m$ the row's `_momentum`, initialized to the first $v$:

$$
\hat{v} = v + \alpha\,(v - m), \qquad m \leftarrow (1-\beta)\,v + \beta\,m
$$

The state lives on the row, so each CFG branch keeps its own.

## Multistep grids

`multistep.py`. One prediction per step, and $m_i = x - \sigma_i\hat{v}$ at the point step $i$ evaluates. `dpm` and `flow_unipc` are deterministic.

### `dpm`

`DpmGrid`, `_dpm_first_order`, `_dpm_second_order`. Steps $i = 0$ and $i = n-1$ take the deterministic `ddim` step $(1-\sigma_{i+1})\,m_i + \sigma_{i+1}\hat{\epsilon}$. The other steps, at `order` 1:

$$
x_{\sigma_{i+1}} = \frac{\sigma_{i+1}}{\sigma_i}\,x + (1-\sigma_{i+1})\left(1 - e^{-h_i}\right)m_i
$$

and at `order` 2:

$$
x_{\sigma_{i+1}} = \frac{\sigma_{i+1}}{\sigma_i}\,x + (1-\sigma_{i+1})\left(1 - e^{-h_i}\right)\left(m_i + \frac{m_i - m_{i-1}}{2\,r_i}\right), \qquad r_i = \frac{h_{i-1}}{h_i}
$$

The history is $(m_{i-1}, \sigma_{i-1})$. Multistep DPM-Solver++(2M) (Lu et al., 2022) in data-prediction form. The order-1 update is the DDIM step; the fixed first and last steps keep $\lambda(1) = -\infty$ and $\lambda(0) = +\infty$ out of the arithmetic.

### `flow_unipc`

`UniPcGrid`, with $B(h) = e^{-h} - 1$ for `solver_type` `bh2` and $B(h) = -h$ for `bh1`. Step $i$ evaluates at the predictor output $\tilde{x}$ (the row's latents) and sets $m_i = \tilde{x} - \sigma_i\hat{v}$.

The corrector (`_corrector`) runs with `use_corrector` from $i = 1$, at the order $p$ of the previous predictor, with $h = h_{i-1}$:

$$
x^c_i = \frac{\sigma_i}{\sigma_{i-1}}\,x^c_{i-1} + (1-\sigma_i)\left(1 - e^{-h}\right)m_{i-1} - (1-\sigma_i)\,B(h)\left(a_1 D + a_2\,(m_i - m_{i-1})\right)
$$

At $p = 1$, $a_2 = \frac{1}{2}$ and there is no $D$ term. At $p = 2$, $D = (m_{i-2} - m_{i-1})/r$ with $r = (\lambda(\sigma_{i-2}) - \lambda(\sigma_{i-1}))/h$, and $(a_1, a_2)$ solves

$$
\begin{pmatrix} 1 & 1 \\ r & 1 \end{pmatrix}\begin{pmatrix} a_1 \\ a_2 \end{pmatrix} = \begin{pmatrix} b_1 \\ b_2 \end{pmatrix}, \qquad b_k = \frac{k!\,\psi_k}{B(h)}, \qquad \psi_1 = \frac{e^{-h} - 1}{-h} - 1, \qquad \psi_2 = \frac{\psi_1}{-h} - \frac{1}{2}
$$

Without the corrector, $x^c_i = \tilde{x}$. The predictor (`_predictor`) has order $p_i = \min(\texttt{order},\ i + 1,\ n - i)$, the last term only with `lower_order_final`:

$$
x_{\sigma_{i+1}} = \frac{\sigma_{i+1}}{\sigma_i}\,x^c_i + (1-\sigma_{i+1})\left(1 - e^{-h_i}\right)m_i - [p_i = 2]\,(1-\sigma_{i+1})\,B(h_i)\,\frac{m_{i-1} - m_i}{2\,r_i}, \qquad r_i = \frac{\lambda(\sigma_{i-1}) - \lambda(\sigma_i)}{h_i}
$$

The history is the last `order` pairs $(m, \sigma)$, $x^c_{i-1}$ and its predictor order. UniPC (Zhao et al., 2023) in data-prediction form: the corrector reuses the call made at $\tilde{x}$, so it costs no model call, and the order-2 predictor uses the reference implementation's fixed coefficient $\frac{1}{2}$. With `bh2` the order-2 predictor equals DPM-Solver++(2M).

### `sa`

`SaGrid`, `sa_update`. The grid is $\sigma_0 \leftarrow \min(\sigma_0, t_0)$ with $t_0$ = `initial_time` and, with `adjust_penultimate_time`, $\sigma_{n-1} \leftarrow \sigma_{n-2}/2$; the row's σ is set to $\sigma_0$ and $x$ is kept. $\tau_i = \eta$ when `stochasticity_start` $\le \sigma_{i+1} \le$ `stochasticity_end`, else 0, and $\xi_i$ is drawn only when $\tau_i > 0$.

The loop: $m_0 = x - \sigma_0\hat{v}(x, \sigma_0)$. For $i = 0, \dots, n-2$, the predictor gives $\tilde{x} = U_{\text{pred}}$ (order 1 at $i = 0$, order 2 after), the model gives $\tilde{m}_{i+1} = \tilde{x} - \sigma_{i+1}\hat{v}(\tilde{x}, \sigma_{i+1})$, and for $i \ge 1$ the corrector gives $x_{\sigma_{i+1}} = U_{\text{corr}}$ with the same $\xi_i$ (else $x_{\sigma_{i+1}} = \tilde{x}$); then $m_{i+1} = \tilde{m}_{i+1}$. The result is $m_{n-1}$, placed at $\sigma_n$: $n$ model calls, the last interval closed by the data prediction instead of a step.

`sa_update` over $\sigma_i \to \sigma_{i+1}$, with $h = h_i$, $\zeta = 1 + \tau_i^2$ and two interpolation nodes $(\lambda_a, m_a)$, $(\lambda_b, m_b)$:

$$
U = e^{-\tau_i^2 h}\,\frac{\sigma_{i+1}}{\sigma_i}\,x_{\sigma_i} + \zeta\,\sigma_{i+1}\,e^{-\tau_i^2\lambda(\sigma_{i+1})}\left(c_a\,m_a + c_b\,m_b\right) + \sigma_{i+1}\sqrt{\max\left(1 - e^{-2\tau_i^2 h},\ 0\right)}\;\xi_i
$$

The predictor's nodes are $(\lambda(\sigma_i), m_i)$ and $(\lambda(\sigma_{i-1}), m_{i-1})$; at order 1 it keeps only $c_a = I_0$. The corrector's nodes are $(\lambda(\sigma_{i+1}), \tilde{m}_{i+1})$ and $(\lambda(\sigma_i), m_i)$. The order-2 coefficients integrate the linear interpolant through the two nodes against $e^{\zeta\lambda}$, plus a correction $C$:

$$
I_0 = \frac{e^{\zeta\lambda(\sigma_{i+1})} - e^{\zeta\lambda(\sigma_i)}}{\zeta}, \qquad I_1 = \frac{e^{\zeta\lambda(\sigma_{i+1})}\left(\zeta\lambda(\sigma_{i+1}) - 1\right) - e^{\zeta\lambda(\sigma_i)}\left(\zeta\lambda(\sigma_i) - 1\right)}{\zeta^2}
$$

$$
c_a = \frac{I_1 - \lambda_b I_0}{\lambda_a - \lambda_b} + C, \qquad c_b = \frac{\lambda_a I_0 - I_1}{\lambda_a - \lambda_b} - C
$$

$$
C_{\text{pred}} = \frac{e^{\zeta\lambda(\sigma_{i+1})}}{h_{i-1}}\left(\frac{h^2}{2} - \frac{\zeta h - 1 + e^{-\zeta h}}{\zeta^2}\right), \qquad C_{\text{corr}} = e^{\zeta\lambda(\sigma_{i+1})}\left(\frac{h}{2} - \frac{\zeta h - 1 + e^{-\zeta h}}{\zeta^2 h}\right)
$$

Second-order SA-Solver (Xue et al., 2023), predict-evaluate-correct in data-prediction form, as in the AWM SD3.5 recipe; $C$ is the few-step coefficient correction of the authors' implementation. It has no per-step density, so GRPO cannot train on it.
