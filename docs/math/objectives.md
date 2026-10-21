# Training objectives

What each trainer minimizes, in the notation of [notation.md](notation.md). Config fields, hooks and the loop are in [../training/trainers.md](../training/trainers.md), rollout programs in [../training/rollouts.md](../training/rollouts.md), variants in [../adapters/variants.md](../adapters/variants.md). Code paths are under `flow_control/training/` unless stated.

Each section defines the loss $\ell_j$ of one train item $j$; [Updates](#updates) turns the items' losses into one optimizer step. $\frac{1}{ND}\lVert\cdot\rVert^2$ is the per-sample MSE (`.mean()` over `[1, N, D]`). $\mathrm{mean}_S$ and $\mathrm{std}_S$ run over a set $S$ of rollouts, and std is the population std (`correction=0`). The draws of training (σ, ε, item order) use the global RNG, seeded per rank ([seeds](../training/trainers.md#seeds)).

## Training points

`TrainingPredictionMixin.noised_point` (`mixins/prediction.py`) re-noises clean latents $x_0$ (fp32) at σ with a fresh ε:

$$
x_\sigma = (1-\sigma)\,x_0 + \sigma\,\epsilon, \qquad v = \epsilon - x_0
$$

`predict_training` evaluates `train_predictor` there: $\hat{v}_\theta$ under the active `default` variant, $\hat{v}_{\text{old}}$ and $\hat{v}_{\text{ref}}$ under `old` and `ref`. A `cfg` train_predictor makes them guided velocities. SFT and the endpoint trainers train on such points; GRPO replays recorded steps instead.

## SFT

`SftTrainer.train_step` (`sft.py`). Each row draws its own σ ~ p(σ) (`timestep_weighting`) and ε. With probability $p_{\text{drop}}$ the row is replaced by its negative row (`processor.get_negative_row`), so it trains on c̄; a processor without a negative row keeps c and warns once. The dropped row keeps `row["negative"]`, so a guided train_predictor still finds c̄.

$$
\ell_j = w(\sigma_j)\,\frac{1}{ND}\left\lVert \hat{v}_\theta(x_{\sigma_j}, \sigma_j, c_j) - v_j \right\rVert^2
$$

Padding rows are forwarded with weight 0 ([SFT updates](#sft-updates)).

### Timestep distributions

`timestep_weighting_registry` (`weighting.py`), used by SFT's `timestep_weighting` and by the `continuous` train timesteps. $u \sim U[0, 1)$ and $\xi \sim N(0, 1)$; $f$ is the rational shift curve of [sampling.md](sampling.md#shift-curves) with $s$ = the member's `shift`; $a$ = `mean`, $b$ = `std`.

**`uniform`**: $\sigma = u$.

**`logit_normal`** (the SFT default; SD3's logit-normal sampling, Esser et al., 2024): $\sigma = \mathrm{expit}(a + b\,\xi)$.

**`mode`** (SD3's mode sampling), with $c$ = `scale`:

$$
\sigma = 1 - u - c\left(\cos^2\frac{\pi u}{2} - 1 + u\right)
$$

It is monotone in $u$, and so within [0, 1], for $-1 \le c \le 2/(\pi - 2)$; the code does not check the range.

**`power_law`**, RAM's sampler, with α = `alpha`:

$$
\sigma = f\left(u^{1/(\alpha+1)}\right)
$$

Before the shift the density is $(\alpha+1)\,\sigma^\alpha$. The RAM reference draws rank $k$ of the 1000-point scheduler grid with weight $(1 - k/1000)^\alpha$ and trains at that rank's shifted σ; this is its continuous limit when $s$ repeats the rollout sampler's shift factor (SD3.5: 3). The default $s = 1$ does not shift.

**`shifted_uniform`**: $\sigma = f(u)$, with density $p(\sigma) = s / (s + (1-s)\,\sigma)^2$.

**`shifted_logit_normal`**: $\sigma = \mathrm{expit}(a + \ln s + b\,\xi) = f(\mathrm{expit}(a + b\,\xi))$.

**`plateau_logit_normal`**: the logit-normal density $p_{\text{LN}}$ with location $a' = a + \ln s$ and scale $b$ up to its mode $\sigma^{\ast}$, constant above it:

$$
p(\sigma) \propto \begin{cases} p_{\text{LN}}(\sigma) & \sigma \le \sigma^{\ast} \\ p_{\text{LN}}(\sigma^{\ast}) & \sigma > \sigma^{\ast} \end{cases}
$$

$\sigma^{\ast}$ solves $\mathrm{logit}(\sigma) - a' - b^2(2\sigma - 1) = 0$ by 20 Newton steps from $\mathrm{expit}(a')$, each clamped to $[10^{-6}, 1 - 10^{-6}]$. With $\Phi$ the standard normal CDF, $P^{\ast} = \Phi((\mathrm{logit}\,\sigma^{\ast} - a')/b)$, $p^{\ast} = p_{\text{LN}}(\sigma^{\ast})$ and $Z = P^{\ast} + (1 - \sigma^{\ast})\,p^{\ast}$, the sample is the inverse CDF:

$$
\sigma = \begin{cases} \mathrm{expit}\left(a' + b\,\Phi^{-1}\left(\mathrm{clamp}(uZ,\ 10^{-6},\ 1 - 10^{-6})\right)\right) & uZ \le P^{\ast} \\ \mathrm{clamp}\left(\sigma^{\ast} + (uZ - P^{\ast})/p^{\ast},\ 0,\ 1\right) & \text{otherwise} \end{cases}
$$

The three `shifted`/`plateau` members are the training distributions of BFL's FLUX.2 representation-comparison blog.

### Loss weights

`loss_weighting_registry` (`weighting.py`), SFT's `loss_weighting`:

| `type` | w(σ) |
|---|---|
| `uniform` (default) | 1 |
| `sigma_squared` | σ⁻², unbounded as σ → 0 |
| `cosmap` | 2 / (π (1 − 2σ + 2σ²)), the density of SD3's CosMap sampling, used as a weight |

## Rewards to advantages

### Reward columns

A rollout program writes the raw rewards (`row["reward_raw"]`) and the normalized ones $r_b$ (`row["reward"]`, each scorer's `normalize` applied: [rewards](../rewards/rewards.md#normalization)), both $[C]$ on `_reward_labels()`. ω is `_reward_weights()`, by default `rollout.weights`: a scorer column weighs the scorer's `weight` times its component weight (`Scorers.weights`). The advantage reads the normalized columns.

`round_robin` (Pref-GRPO, Wang et al., 2025) puts a win rate $r_{b,\text{rr}}$ first, weighted by its `weight` and not normalized. `preferences` (`rollouts/pairwise.py`) asks the judge for $s_{ab}$ = P(a ≻ b); with `swap_orders` it asks both orders and folds out the position bias:

$$
s(a, b) = \frac{s_{ab} + 1 - s_{ba}}{2}
$$

`win_rates` scores rollout b over the $n_b$ games of its `schedule`, with $s(b, a) = 1 - s(a, b)$ when b played second:

$$
r_{b,\text{rr}} = \frac{\frac{1}{2} + \sum_{\text{games } (b, a)} s(b, a)}{n_b + 1}
$$

The extra half game is a tie with itself, so over all K(K − 1)/2 pairs this is the row mean of the win matrix with ½ on the diagonal.

### Advantage estimators

`RolloutMixin._compute_advantages` (`mixins/rollout.py`) gathers $r_b$ of all B = M·K rollouts from every rank and groups them by prompt, so $b' \sim b$ runs over the K rollouts of b's prompt. With the combined reward

$$
R_b = \sum_{c=1}^{C} \omega_c\, r_{b,c}
$$

each `advantage_registry` member (`advantage.py`) computes $A_b$, with $\epsilon_A$ = `eps` ($10^{-4}$).

**`per_prompt`** (default). With `use_global_std` (default) the denominator is the std over all B rollouts; without it, the group's own std (GRPO, Shao et al., 2024):

$$
A_b = \frac{R_b - \mathrm{mean}_{b' \sim b}\,R_{b'}}{\mathrm{std}_B\,R + \epsilon_A} \qquad \text{or} \qquad A_b = \frac{R_b - \mathrm{mean}_{b' \sim b}\,R_{b'}}{\mathrm{std}_{b' \sim b}\,R_{b'} + \epsilon_A}
$$

**`global`**:

$$
A_b = \frac{R_b - \mathrm{mean}_B\,R}{\mathrm{std}_B\,R + \epsilon_A}
$$

**`gdpo`** (GDPO): each column is normalized within its group before the weights combine them, so a column with a wide spread cannot drown the others; the sum is then normalized over the batch:

$$
r'_{b,c} = \frac{r_{b,c} - \mathrm{mean}_{b' \sim b}\,r_{b',c}}{\mathrm{std}_{b' \sim b}\,r_{b',c} + \epsilon_A}, \qquad R'_b = \sum_{c} \omega_c\,r'_{b,c}, \qquad A_b = \frac{R'_b - \mathrm{mean}_B\,R'}{\mathrm{std}_B\,R' + \epsilon_A}
$$

$A_b$ is not clipped here. Each loss clamps it with its own `adv_clip_max`, $\tilde{A}_b = \mathrm{clamp}(A_b, -A_{\max}, A_{\max})$. Every rank keeps the slice of its own rollouts.

### Rollout metrics

`_compute_advantages` logs, over the B gathered rollouts:

- `rollout/reward_mean`, `rollout/reward_std`: $\mathrm{mean}_B\,R$ and $\mathrm{std}_B\,R$.
- `rollout/reward_group_std`: the mean over the M prompts of $\mathrm{std}_{b' \sim b}\,R_{b'}$, the spread a group-relative advantage sees. A collapse of rollout diversity shows here while the batch std holds.
- `rollout/adv_abs_mean`: $\mathrm{mean}_B\,|A_b|$, before clipping.
- `rollout/raw/<label>_{mean,std,group_std}` and `rollout/normalized/<label>_{mean,std}`: the same statistics of one unweighted column of `reward_raw` or `reward`.

`_finish_rollouts` logs `rollout/policy_lag`: the training epoch minus the epoch whose weights sampled the batch.

## GRPO

`GrpoTrainer` (`grpo.py`), Flow-GRPO (Liu et al., 2025).

### Recorded steps

The rollouts are sampled under `default`. The `record` observer keeps every stochastic step $i$ of a rollout ($\eta \ne 0$ and $0 < \sigma_{i+1} < \sigma_i$, [transitions](sampling.md#transitions)): $x_{\sigma_i}$, $\sigma_i$, $x_{\sigma_{i+1}}$, and the step's [log-density](sampling.md#log-density) under the rollout's own prediction, $\log p_{\theta_{\text{old}}}(x_{\sigma_{i+1}} \mid x_{\sigma_i})$. Under an [`sde_window_grid`](sampling.md#sde_window_grid) only the window's steps are stochastic. $S_b$ is the set of recorded steps of rollout b; a rollout with none raises (`_check_rollouts`). The train items are the pairs $(b, i)$, $i \in S_b$; every inner epoch takes all of them, the rollouts shuffled and each rollout's steps contiguous and in order.

### Loss

`_replay` recomputes each step's log-density with v̂ = `train_predictor` at $(x_{\sigma_i}, \sigma_i)$ under θ, so train_predictor must repeat the rollout's prediction for ρ = 1 at θ = θ_old. With $\mu_i^\theta$ and $\mu_i^{\text{ref}}$ the step means under θ and θ_ref (`ref` replayed under `torch.no_grad()`) and $g_i$ = `StepLogProb.std_dev`, `grpo_loss` computes

$$
\rho_{b,i} = \exp\left(\log p_\theta(x_{\sigma_{i+1}} \mid x_{\sigma_i}) - \log p_{\theta_{\text{old}}}(x_{\sigma_{i+1}} \mid x_{\sigma_i})\right)
$$

$$
\ell_{b,i} = \max\left(-\tilde{A}_b\,\rho_{b,i},\ -\tilde{A}_b\,\mathrm{clamp}(\rho_{b,i},\ 1 - \epsilon_{\text{clip}},\ 1 + \epsilon_{\text{clip}})\right) + \beta_{\text{KL}}\,\frac{1}{2g_i^2}\cdot\frac{1}{ND}\left\lVert \mu_i^\theta - \mathrm{sg}[\mu_i^{\text{ref}}] \right\rVert^2
$$

The log-densities are means over the N·D elements, so ρ is the exponential of a per-element log-ratio, and the gradient reaches θ through $\mu_i^\theta$ only. The recorded log-density is a constant: under `rollout_lookahead` N it comes from the weights N epochs back ([n-step-off](../training/rollouts.md#n-step-off-1)).

The KL term divides by $g_i$, not by the step std $\delta_i$, as Flow-GRPO's reference code does. For [`euler`](sampling.md#euler) $\delta_i = g_i\sqrt{-\Delta\sigma_i}$, so the term is $-\Delta\sigma_i$ times the per-element KL between the step Gaussians $N(\mu_i^\theta, \delta_i^2)$ and $N(\mu_i^{\text{ref}}, \delta_i^2)$; for the other transitions $g_i = \delta_i$ and it is that KL.

Defaults: $\epsilon_{\text{clip}}$ = `clip_range` = $10^{-4}$, $A_{\max}$ = 5, $\beta_{\text{KL}}$ = 0; `ref` is required only when $\beta_{\text{KL}} > 0$. Logged as means over the update's items: `approx_kl` $= \frac{1}{2}(\log p_\theta - \log p_{\theta_{\text{old}}})^2$, `clipfrac` $= \mathbf{1}[|\rho - 1| > \epsilon_{\text{clip}}]$, `ratio_mean`.

## Endpoint trainers

`EndpointTrainer` (`endpoint.py`). The presets `nft`, `awm` and `ram` differ only in their defaults ([trainers](../training/trainers.md#another-objective-on-an-endpoint-trainer)).

### Train points

A rollout is sampled under `objective.rollout_variant()`. Its $x_0$ is the sampled latents (`clean_latents` after `decode`), and the `keep_sigmas` observer records its executed grid $\sigma_0 > \dots > \sigma_n$, sequential grids concatenated. Per inner epoch, `_build_train_plan` shuffles the rollouts and draws each one's training σ with `train_timesteps.draw`; each σ becomes a [training point](#training-points) with a fresh ε, unrelated to the latents the rollout passed through. `_loss_batched` evaluates the `required_variants()` under `torch.no_grad()`, so $\hat{v}_{\text{old}}$ and $\hat{v}_{\text{ref}}$ are constants, then $\hat{v}_\theta$, and `objective.compute` returns $\ell_j$ with the rollout's unclipped $A_b$.

**`grid`** (`GridTimesteps`) takes step starts $\sigma_i$, $i < n$. With $i_0 = 1$ under `exclude_first` (else 0):

$$
i_1 = \max\left(i_0 + 1,\ \lfloor n \cdot \mathrm{window} \rfloor\right), \qquad k = \begin{cases} \min(\mathrm{count},\ i_1 - i_0) & \text{count set} \\ \max\left(1,\ \lfloor (i_1 - i_0) \cdot \mathrm{fraction} \rfloor\right) & \text{otherwise} \end{cases}
$$

It raises when $i_1 > n$. `random` takes $k$ distinct indices of $[i_0, i_1)$ uniformly. `stratified` takes one per bin, with edges $e_t = i_0 + t\,(i_1 - i_0)/k$ and $u_t \sim U[0, 1)$:

$$
i^{(t)} = \lfloor e_t \rfloor + \left\lfloor u_t\left(\lfloor e_{t+1} \rfloor - \lfloor e_t \rfloor\right) \right\rfloor, \qquad t = 0, \dots, k-1
$$

The count depends only on $n$, so ranks whose rollouts have equal step counts submit equal item counts.

**`continuous`** (`ContinuousTimesteps`): `count` independent draws from `weighting` ([timestep distributions](#timestep-distributions), default `logit_normal`), ignoring the grid.

### Variants per objective

| Objective | Samples the rollouts (`rollout_variant`) | Reads (`required_variants`) |
|---|---|---|
| `nft` | `old` | `old`; `ref` when `kl_beta` > 0 |
| `awm` | `default`; `old` with `off_policy` | `ref` when `beta` > 0; `old` when `ema_beta` > 0 or `off_policy` |
| `ram` | `old` | `old`, `ref` |
| `weighted_fm` | `default` | none |

Under n-step-off a batch is sampled by the `old` of its launch epoch while the loss reads the current one ([n-step-off](../training/rollouts.md#n-step-off-1)).

### `nft`

`NftObjective` (`nft.py`), DiffusionNFT (Zheng et al., 2025, arXiv:2509.16117), with β = `beta`, and $A_{\max}$ and $\beta_{\text{KL}}$ the objective's `adv_clip_max` and `kl_beta`. `adv_mode` then reshapes $\tilde{A}_b$: `positive_only` keeps $\max(\tilde{A}_b, 0)$, `negative_only` $\min(\tilde{A}_b, 0)$, `binary` $\mathrm{sign}(\tilde{A}_b)$, `all` (default) all of it. The result maps to the soft label

$$
y_b = \mathrm{clamp}\left(\frac{\tilde{A}_b}{2A_{\max}} + \frac{1}{2},\ 0,\ 1\right)
$$

so `binary` gives $y_b = \frac{1}{2} \pm \frac{1}{2A_{\max}}$ (0.6 and 0.4 at the default 5). The current and old velocities define the implicit positive and negative policies, each read as an $\hat{x}_0$:

$$
v^+ = \beta\,\hat{v}_\theta + (1-\beta)\,\hat{v}_{\text{old}}, \qquad v^- = (1+\beta)\,\hat{v}_{\text{old}} - \beta\,\hat{v}_\theta, \qquad \hat{x}_0^\pm = x_\sigma - \sigma\,v^\pm
$$

Each is regressed on $x_0$ with a per-sample adaptive weight:

$$
L^\pm = \frac{\frac{1}{ND}\lVert \hat{x}_0^\pm - x_0 \rVert^2}{\mathrm{sg}\left[\max\left(\frac{1}{ND}\lVert \hat{x}_0^\pm - x_0 \rVert_1,\ 10^{-5}\right)\right]}
$$

$$
\ell_j = \frac{A_{\max}}{\beta}\left(y_b\,L^+ + (1 - y_b)\,L^-\right) + \beta_{\text{KL}}\,\frac{1}{ND}\lVert \hat{v}_\theta - \hat{v}_{\text{ref}} \rVert^2
$$

Since $\hat{x}_0^\pm - x_0 = \sigma\,(v - v^\pm)$, $L^\pm$ is the velocity error of $v^\pm$ times σ over its mean absolute value. A positive advantage pulls $v^+$, and with it $\hat{v}_\theta$, toward the sample's $v$; a negative one pulls $v^-$ toward it, which moves $\hat{v}_\theta$ to the mirror side of $\hat{v}_{\text{old}}$. The factor $A_{\max}/\beta$ is the reference implementation's. Defaults: β = 1, $\beta_{\text{KL}}$ = 0.01, $A_{\max}$ = 5. Also logged: `old_deviate` $= \frac{1}{ND}\lVert \hat{v}_\theta - \hat{v}_{\text{old}} \rVert^2$.

### `awm`

`AwmObjective` (`awm.py`), AWM (Advantage Weighted Matching, arXiv:2509.25050). With $e_\phi = \frac{1}{ND}\lVert \hat{v}_\phi - v \rVert^2$ for $\phi \in \lbrace \theta, \text{old} \rbrace$, $\epsilon_0 = 10^{-10}$ and $p$ = `ghuber_power` (0.25), `weighting` (default `ghuber`) picks the flow-matching log-likelihood surrogate:

$$
\log\tilde{p}_\phi = \begin{cases} -e_\phi & \text{uniform} \\ -\sigma\,e_\phi & \text{t} \\ -\sigma^2 e_\phi & \text{t2} \\ -\sigma\left(\sqrt{e_\phi + \epsilon_0} - \sqrt{\epsilon_0}\right) & \text{huber} \\ -\sigma\left((e_\phi + \epsilon_0)^p - \epsilon_0^p\right)/p & \text{ghuber} \end{cases}
$$

The ratio's denominator is $\log\tilde{p}_{\text{old}}$ with `off_policy`, and $\log\tilde{p}_\theta$ itself otherwise, so on-policy ρ = 1 in value and $\nabla\rho = \nabla\log\tilde{p}_\theta$:

$$
\rho_j = \exp\left(\log\tilde{p}_\theta - \mathrm{sg}[\log\tilde{p}_{\text{old}}]\right)
$$

With $a$ = `advantage_max` (1) rescaling the clamped advantage to $[-a, a]$, β = `beta`, $\beta_{\text{ema}}$ = `ema_beta`, and $\epsilon_{\text{clip}}$ and $A_{\max}$ the objective's `clip_range` and `adv_clip_max`:

$$
\ell_j = \max\left(-\frac{a\tilde{A}_b}{A_{\max}}\,\rho_j,\ -\frac{a\tilde{A}_b}{A_{\max}}\,\mathrm{clamp}(\rho_j,\ 1 - \epsilon_{\text{clip}},\ 1 + \epsilon_{\text{clip}})\right) + \beta\,D(\hat{v}_\theta, \hat{v}_{\text{ref}}) + \beta_{\text{ema}}\,D(\hat{v}_\theta, \hat{v}_{\text{old}})
$$

$D$ follows `kl_weight` for the first term and `kl_ema_weight` for the second:

$$
D_{\text{uniform}}(\hat{v}_\theta, \hat{v}_\phi) = \frac{1}{ND}\lVert \hat{v}_\theta - \hat{v}_\phi \rVert^2, \qquad D_{\text{elbo}} = \frac{D_{\text{uniform}}}{2\tilde{g}^2}, \qquad \tilde{g} = 0.7\sqrt{\frac{\sigma}{1 - \min(\sigma, 0.99)}}
$$

$\tilde{g}$ is the `euler` diffusion coefficient $g_i$ at a hard-coded η = 0.7. On-policy the policy term's gradient is $-(a\tilde{A}_b/A_{\max})\,\nabla\log\tilde{p}_\theta$: flow matching weighted by the advantage. Defaults: β = 0.001, $\beta_{\text{ema}}$ = 1, $A_{\max}$ = 5, $\epsilon_{\text{clip}}$ = 1 (ρ inside [0, 2] is not clipped). The reference's TRPO-EMA is an `old` variant with `decay` 0.3 and a `linear_ramp` warmup of `ramp_rate` 0.001 ([warmups](#warmups)). Logged: `ratio_mean`, `clipfrac` $= \mathbf{1}[|\rho - 1| > \epsilon_{\text{clip}}]$.

### `ram`

`RamObjective` (`ram.py`), RAM (Reinforce Adjoint Matching, arXiv:2605.10759), with $a$ = `reward_multiplier` (100). $\tilde{A}_b$ is clamped only when the objective's `adv_clip_max` is set (default: unclipped, as in the reference).

$$
\ell_j = \frac{1}{ND}\left\lVert \hat{v}_\theta - \mathrm{sg}\left[\hat{v}_{\text{ref}} + a\,\tilde{A}_b\,\left(v - \hat{v}_{\text{old}}\right)\right] \right\rVert^2
$$

There is no KL term: the target is anchored at $\hat{v}_{\text{ref}}$, and a larger $a$ regularizes less. Over θ_old's own samples the expected $v$ at $x_\sigma$ is θ_old's velocity, so $v - \hat{v}_{\text{old}}$ averages to about zero and only its correlation with the advantage moves the target off $\hat{v}_{\text{ref}}$. The `ram` preset turns gradient clipping off (`clip_grad_norm: 0.0`), as the reference does. Logged: `target_norm` (the target's MSE norm), `base_deviate` $= \frac{1}{ND}\lVert \hat{v}_\theta - \hat{v}_{\text{ref}} \rVert^2$.

### `weighted_fm`

`WeightedFlowMatchingObjective` (`weighted_fm.py`): flow matching on the policy's own rollouts, weighted by the unclipped advantage, with η = the objective's `eta` (not the SDE strength) and $e_\theta = \frac{1}{ND}\lVert \hat{v}_\theta - v \rVert^2$:

$$
\ell_j = \begin{cases} \exp(\eta A_b)\,e_\theta & \text{exp (default)} \\ \max(1 + \eta A_b,\ 0)\,e_\theta & \text{linear} \end{cases}
$$

η < 0 weights the low-reward rollouts up.

## Variants

### EMA update

`VariantsMixin.update_variants` (`flow_control/adapters/weights.py`) moves every `ema` variant whose `every` matches. With $k$ the variant's update count (`variant_counts`, checkpointed) and $d_k$ = `warmup.get_decay(k, decay)`:

$$
\theta_{\text{ema}} \leftarrow d_k\,\theta_{\text{ema}} + (1 - d_k)\,\theta_{\text{src}}
$$

per tensor, in fp32 (`torch._foreach_lerp_`), and skipped when $d_k \ge 1$; $k$ counts every call. $\theta_{\text{src}}$ is the variant's `of` (default `default`), and $\theta_{\text{ema}}$ starts as an fp32 copy of it at setup. `optimizer_step` moves the `every: "step"` EMAs after `optimizer.step()`; the RL loop moves the `every: "epoch"` ones (usually θ_old) after each epoch's training phase. For a LoRA source the tensors are the factors A and B and the `extra_trainable` clones, so the EMA's weight delta is the product of the averaged factors, not the average of the products.

### Warmups

`warmup_registry` (`flow_control/adapters/variants.py`), with $d$ = `decay`:

- **`none`** (default): $d_k = d$.
- **`standard`**: $d_k = \min\left((1 + k)/(10 + k),\ d\right)$, 0.1 at the first update.
- **`linear_ramp`**, with $k_0$ = `flat_steps` and $a$ = `ramp_rate`: $d_k = 0$ for $k < k_0$, else $\min\left((k - k_0)\,a,\ d\right)$. $d_k = 0$ copies the source.

## Updates

### Rollout trainer updates

`RolloutTrainerBase._train_on_rollouts` (`rollout_trainer.py`) with `MicrobatchTrainMixin.iter_micro_updates` (`mixins/microbatch.py`). Each rank cuts an inner epoch's items into chunks $U$ of $G/W$ items, the last possibly shorter (every rank holds the same count, checked by an `all_reduce`), and each chunk into microbatches $J$ of `train_micro_batch_size`. `_loss_batched` returns the mean of $\ell_j$ over $J$, its backward is scaled by $|J|/|U|$ (`loss_scale`), and FSDP averages the gradients over the $W$ ranks, so one optimizer step takes the gradient of

$$
\mathcal{L} = \frac{1}{W}\sum_{w=1}^{W}\sum_{J \subset U_w}\frac{|J|}{|U_w|}\cdot\frac{1}{|J|}\sum_{j \in J}\ell_j = \frac{1}{W\,|U|}\sum_{w=1}^{W}\sum_{j \in U_w}\ell_j
$$

the mean over the update's items. When $G$ covers an inner epoch's items (GRPO: every recorded step of every rollout) there is one step per inner epoch.

### SFT updates

`SftTrainer.run` and `train_step`. A rank accumulates $n_{\text{acc}}$ = `grad_acc_steps` $= G/W/$`train_micro_batch_size` microbatches $J$, each with $R_J$ real (non-padding) rows and backward scaled by $1/n_{\text{acc}}$:

$$
\mathcal{L} = \frac{1}{W\,n_{\text{acc}}}\sum_{w=1}^{W}\sum_{J}\frac{1}{\max(1, R_J)}\sum_{j \in J,\ j\ \text{real}}\ell_j
$$

Padding rows keep FSDP's collectives balanced and weigh 0, so a real row of a partially padded microbatch weighs $1/R_J$ instead of $1/|J|$. Without padding $\mathcal{L}$ is the mean over the $G$ rows.

### Optimizer step

`optimizer_step` (`mixins/base.py`), with $c_{\max}$ = `clip_grad_norm` > 0 and the norm over all of the transformer's gradients:

$$
\nabla_\theta\mathcal{L} \leftarrow \nabla_\theta\mathcal{L}\cdot\min\left(1,\ \frac{c_{\max}}{\lVert\nabla_\theta\mathcal{L}\rVert + 10^{-6}}\right)
$$

then `optimizer.step()`, the `every: "step"` [EMA update](#ema-update), `scheduler.step()`.

## Contrib: `vae`

`VaeTrainer` (`flow_control/contrib/rgba_vae_training/`), the AlphaVAE losses (`RGBAVAELoss`, `loss.py`). It trains a VAE on images, not latents: the target $I$ (`target_image`) is BCHW in $[-1, 1]$, with color $I_{\text{rgb}}$ and alpha $\alpha = (I_{\text{a}} + 1)/2$; $\hat{I}$ = `decode(z)` with $z$ sampled from the posterior $q(z \mid I)$. With probability `blend_prob` the target is first composited over a background whose channels are drawn from $\lbrace 0, \frac{1}{2}, 1 \rbrace$ (in [0, 1]), with α set to 1. $L_{\text{rec}}$, $L_{\text{p}}$ and the KL terms are summed over a sample's elements and averaged over the batch (`_reduce`).

Reconstruction, with $\Delta p = I_{\text{rgb}}\alpha - \hat{I}_{\text{rgb}}\hat{\alpha}$, $\Delta\alpha = \alpha - \hat{\alpha}$ and the per-channel constants $E_b$, $E_{b^2}$ (`custom_Eb`, `custom_Eb2`):

$$
L_{\text{rec}} = \sum\left(\Delta p^2 - 2E_b\,\Delta p\,\Delta\alpha + E_{b^2}\,\Delta\alpha^2\right)
$$

It is the expected squared error of both images composited over a random background $b$ with $\mathbb{E}[b] = E_b$ and $\mathbb{E}[b^2] = E_{b^2}$, whose composite difference is $\Delta p - b\,\Delta\alpha$. `use_naive_mse` uses $\sum (\hat{I} - I)^2$ instead.

Perceptual, on two composites:

$$
L_{\text{p}} = \frac{1}{2}\left(\mathrm{LPIPS}\left(I_{\text{rgb}}\alpha,\ \hat{I}_{\text{rgb}}\hat{\alpha}\right) + \mathrm{LPIPS}\left(I_{\text{rgb}}\alpha + 1 - \alpha,\ \hat{I}_{\text{rgb}}\hat{\alpha} + 1 - \hat{\alpha}\right)\right)
$$

In $[-1, 1]$ the first composite lies over 0, mid-gray; the second over white.

Prior, $L_{\text{KL}} = \mathrm{KL}\left(q(z \mid I)\ \Vert\ N(0, 1)\right)$ summed over the latent elements. With `ref_vae` and `ref_kl_scale`, the posteriors of the composites over black and white, $I^{\text{k}} = (I_{\text{rgb}}\alpha - (1 - \alpha),\ 1)$ and $I^{\text{w}} = (I_{\text{rgb}}\alpha + 1 - \alpha,\ 1)$, are pulled to those of the 3-channel reference encoder $q_{\text{ref}}$ on their color channels:

$$
L_{\text{ref}} = \frac{1}{2}\sum_{x \in \lbrace I^{\text{k}}, I^{\text{w}} \rbrace}\mathrm{KL}\left(q(z \mid x)\ \Vert\ \mathrm{sg}[q_{\text{ref}}(z \mid x_{\text{rgb}})]\right)
$$

Adversarial, from optimizer step `gan_start_step` on, with the PatchGAN discriminator $D$ and $w_{\text{out}}$ = `decoder.conv_out.weight`:

$$
L_G = -\tau\,\mathrm{mean}\,D(\hat{I}), \qquad \tau = \mathrm{sg}\left[\mathrm{clamp}\left(\frac{\lVert\nabla_{w_{\text{out}}} L_{\text{rec}}\rVert}{\lVert\nabla_{w_{\text{out}}}\,\mathrm{mean}\,D(\hat{I})\rVert + 10^{-4}},\ 0,\ 10^4\right)\right]
$$

$$
L_D = \frac{1}{2}\left(\mathrm{mean}\max\left(0,\ 1 - D(I)\right) + \mathrm{mean}\max\left(0,\ 1 + D(\mathrm{sg}[\hat{I}])\right)\right)
$$

The generator loss is $L_{\text{rec}}$ plus `lpips_scale` $L_{\text{p}}$, `kl_scale` $L_{\text{KL}}$, `ref_kl_scale` $L_{\text{ref}}$ and `generator_loss_weight` $L_G$; a null scale drops its term. $L_D$, times `discriminator_loss_weight`, trains $D$ with its own optimizer from the same step. A microbatch is one row: its losses backward times $W/G$, padding rows times 0, so unlike SFT the padding rows count in the divisor.
