# Notation

The symbols of every formula in `math/` and the code names they stand for. [sampling.md](sampling.md) applies them to the sampler nodes, [objectives.md](objectives.md) to the trainers. Add a symbol here before using it in either.

## Conventions

- **Interpolation.** $x_\sigma = (1-\sigma)\,x_0 + \sigma\,\epsilon$ for $\sigma \in [0, 1]$, so $\sigma = 1$ is pure noise. The velocity is $v = \epsilon - x_0 = dx_\sigma / d\sigma$. A subscript on $x$ is always a noise level: $x_0$ is the clean sample and $x_{\sigma_i}$ the latents at grid point $i$, never a step index.
- **Estimates.** A velocity $\hat{v}$ at $(x, \sigma)$ gives $\hat{x}_0 = x - \sigma\hat{v}$ and $\hat{\epsilon} = x + (1-\sigma)\,\hat{v}$. Then $x = (1-\sigma)\,\hat{x}_0 + \sigma\hat{\epsilon}$ and $\hat{v} = \hat{\epsilon} - \hat{x}_0$, and $-\hat{\epsilon}/\sigma$ estimates the score of $x_\sigma$.
- **Time direction.** Sampling runs from $\sigma = 1$ to $0$: $\sigma_{i+1} < \sigma_i$ and $\Delta\sigma_i < 0$. A grid whose `sigma_end` lies above the row's σ (inversion) runs upward; a formula that needs a downward step says so.
- **Shapes.** A sample's latents are packed `[1, N, D]`: N tokens of D channels. Products are elementwise. Sums, means and norms run over all N·D elements of one sample unless stated; $\lVert\cdot\rVert_D$ is the per-token norm over the D channels.
- **Families.** Adapters translate each family's time scale and prediction target into σ and v ([framework conventions](../adapters/adapter.md#framework-conventions)), so every formula holds for every family. A flow-map adapter (FLUX.3 distilled) also reads $\sigma_{i+1}$ (`row["sigma_next"]`): its $v_\theta$ is the average velocity over the step and exists only inside a transition.
- **Noise.** $\epsilon$ and $\xi$ are standard normal. $\xi$ is a fresh draw at every use, from the row's generator (`randn(row)` on `row["_generator"]`); a node that reuses a draw says so.
- **Local symbols.** A node's config field appears as the Greek letter it is named after (`eta` η, `alpha` α, `beta` β) or in code font. Such letters, and symbols defined next to a single formula, are local to that formula.

## Points and velocities

| Symbol | Meaning | Code |
|---|---|---|
| x₀ | clean latents | `clean_latents`, `x0` |
| ε | the noise paired with x₀: a sample's initial latents at σ = 1, or a training draw | `processor.initialize_latents`; `point["noise"]` |
| ξ | a fresh standard-normal draw inside a node | `randn(row)` |
| σ | noise level in [0, 1] | `row["sigma"]`, `sigma`; `timestep` (adapter input); `t` (`sigma_like`, SA-Solver `times`) |
| x_σ, x | latents at σ; x alone is the row's current latents | `row["noisy_latents"]`, `x`, `xt`, `sample` |
| x_src | latents of a condition image selected by `from` | `ConditionImage.latents(row)` |
| v | the flow-matching target ε − x₀ | `target`, `noise - clean_latents` |
| v_θ(x, σ, c) | model velocity under weights θ | `predict_velocity_batched`; the `model` prediction |
| v̂ | the velocity a prediction tree returns at the row's point; the one a step uses | `velocity`, `StepFrame.velocity`; `current`, `aux[name]` in objectives |
| x̂₀, ε̂ | x − σv̂ and x + (1 − σ)v̂ | `pred_original_sample`, `x0`; `noise_pred`, `noise_estimate` |

## Conditions and guidance

| Symbol | Meaning | Code |
|---|---|---|
| c | the row's conditioning (prompt embeddings, reference latents, ...) | condition keys of the row |
| c̄ | negative conditioning | `row["negative"]`, merged by `Cfg.uncond_row` |
| v_c, v̄ | conditional and negative branch of a guided prediction | `cond`, `neg` |
| γ | guidance scale | `Cfg.scale` |
| v_γ | v̄ + γ(v_c − v̄), with the optional renorm | `Cfg.guide` |
| κ | CFG++ factor | `kappa` |

## Grids and shifts

| Symbol | Meaning | Code |
|---|---|---|
| n | steps of a grid | `steps` |
| i | step index, 0 ≤ i < n | `GridFrame.index` |
| σ₀ > σ₁ > … > σₙ | the executed grid: σ₀ is the row's σ when the grid starts, σₙ is `sigma_end` | `GridFrame.sigmas`, `BaseGrid.sigmas(row)`; `row["sigmas"]` for endpoint trainers |
| Δσᵢ | σᵢ₊₁ − σᵢ | `dt`, `dsigma` |
| u | uniform grid variable in [0, 1] | `u` |
| F, F⁻¹ | shift curve σ = F(u) and its inverse | `Curve.__call__`, `Curve.inverse` |
| f, s | the rational curve f(u) = s·u / (1 + (s − 1)·u) and its factor | `ShiftCurve.f`, `BaseShift.factor` |
| L | image sequence length a shift reads | `seq_len`, `BaseShift._seq_len` |
| n_full | step count of the full 1 → 0 grid, passed to step-dependent shifts | `num_steps` |

## Steps

| Symbol | Meaning | Code |
|---|---|---|
| η | SDE strength | `eta` |
| μᵢ | mean of step i | `mean` |
| δᵢ | std of the step's noise: x_σᵢ₊₁ = μᵢ + δᵢξ | `scale`, the second value of `GaussianTransition.moments` |
| gᵢ | the std GRPO's KL divides by: Euler's diffusion coefficient, δᵢ for the other transitions | `std_dev_t`, `StepLogProb.std_dev` |
| log p_θ(x_σᵢ₊₁ ∣ x_σᵢ) | log-density of step i when v̂ is computed under θ | `GaussianTransition.log_prob`, `StepLogProb.log_prob`; GRPO's recorded `point["log_prob"]` |
| λ(σ) | log((1 − σ)/σ), half the log-SNR | `_lambda` |
| hᵢ | λ(σᵢ₊₁) − λ(σᵢ), positive for a downward step | `h` |
| mᵢ | x̂₀ of step i, a multistep solver's history | `x0`, `m0`, `m1`, `x0s`, `models` |

## Policies

| Symbol | Meaning | Code |
|---|---|---|
| θ | trained weights: the `default` variant | `"default"` |
| θ_old | the policy that sampled the rollouts: the `old` variant (an EMA of θ moved per epoch) where an objective reads it; for GRPO, the weights at sampling time, whose log p is recorded | `"old"`, `aux["old"]`; `point["log_prob"]` |
| θ_ref | reference policy: the `ref` variant, usually an alias of `base` | `"ref"`, `aux["ref"]` |
| θ_base | the loaded weights without trainable deltas | `"base"` |
| θ_ema | an EMA of θ | `"ema"` |
| d | EMA decay: θ_ema ← d·θ_ema + (1 − d)·θ | `decay`, `warmup.get_decay` |

## Training and rewards

| Symbol | Meaning | Code |
|---|---|---|
| p(σ) | distribution of training σ | `timestep_weighting.sample_timesteps` (SFT), `train_timesteps.draw` (endpoint trainers) |
| w(σ) | loss weight of a training σ | `loss_weighting.get_weights` |
| p_drop | probability that SFT trains a row on c̄ | `cfg_drop_prob` |
| M | prompts per epoch | `num_prompts_per_epoch` |
| K | rollouts per prompt (group size) | `num_rollouts_per_prompt` |
| b, B | rollout index in an advantage batch, and its size B = M·K | rows of `rewards` |
| C | reward columns | `_reward_labels()` |
| r_b | reward vector of rollout b, C columns | `row["reward"]` (normalized), `row["reward_raw"]` |
| ω | reward column weights, C values | `rollout.weights`, `_reward_weights()` |
| A_b | advantage of rollout b | `advantages[b]`, `point["advantage"]` |
| A_max | advantage clip | `adv_clip_max` |
| ρ | importance ratio exp(log p_θ − log p_θ_old) | `ratio` |
| ε_clip | ratio clip range | `clip_range` |
| β_KL | KL coefficient | `kl_beta` |
| sg[·] | stop-gradient: the value, with no gradient flowing back through it | `.detach()`, `torch.no_grad()` |
| v̂_θ, v̂_old, v̂_ref | v̂ of `train_predictor` at a training point under θ, θ_old, θ_ref | `current`; `aux["old"]`, `aux["ref"]` |
| ℓ_j | loss of train item j: an SFT row, a GRPO recorded step, or an endpoint re-noised point | `per_sample_losses`, `grpo_loss`, `LossOutput.loss` |
| G, W | items per optimizer update over all ranks, and the number of ranks | `train_batch_size`, `world_size` |
| R_b | combined reward Σ_c ω_c r_b,c | `combined` |
| b′ ∼ b | b′ runs over the K rollouts of b's prompt (same `row[KEY]`), b included | `prompt_ids` |
| ε_A | floor added to the std of an advantage estimator | `eps` |
| Ã_b | A_b clamped to [−A_max, A_max] by a loss | `torch.clamp(adv, -adv_clip_max, adv_clip_max)` |
| y_b | NFT's soft positive label in [0, 1], derived from A_b | `r` in `NftObjective.compute` |
