# Training Report

[简体中文](TRAINING.md) | [English](TRAINING_EN.md)

This document summarizes the released models, training recipe, evaluation protocol and limitations. All numbers are from local runs.

## Model

5.61M action-value Q network:

```text
state (132) ─► MLP(512) ─┐
history (15×33) ─► LSTM(256) ─┴─► z (512)
action ID + 13 features ─► embedding/MLP(128) ─┐
                                               ├─► Q(s,a)
z ─────────────────────────────────────────────┘
```

- Shared trunk for all three seats + role one-hot.
- **26,064 rank-canonical actions + PASS**; no suit distinctions; 54-bit masks.
- Adapter: `z = z + adapter(z)`, 512→1024→512 residual MLP with zero-initialized last layer.

## Training recipe

| Stage | Content | Games |
|---|---|---:|
| v1 oracle | oracle landlord + opponent pool | 0.829M |
| v1 random | random landlord fine-tune | 1.24M |
| v2 | 2048 envs; oracle 15% + random 85%; pool 30% + bot 10% | 6.56M |
| adapter | 1.05M residual adapter, low LR | 1.33M |
| no-shuffle expert | from adapter, no-shuffle deals | 1.31M |

Total regular training: about **9.96M games**; no-shuffle expert: **1.31M** extra.

Key hyperparameters: 2048 parallel games, 16,384 games/iter, 768 train steps, batch 2048, lr 8e-5→4e-5 (adapter 4e-5→2e-5), epsilon 0.05→0.02 (adapter 0.03→0.015), 3-checkpoint soup.

### No-shuffle rule

1. Random first deck;
2. record played groups in actual play order;
3. append the three remaining hands sorted by rank, giving 54 cards;
4. random cut;
5. three consecutive 17-card blocks randomly assigned to seats; last 3 are bottom.

Measured about **1.6 bombs/hand** vs 0.06 for normal random deals. This is a reproducible local approximation, not an official server reimplementation.

## Evaluation

- Regular: 1024 fixed deals, three seats, 2048–3072 games per mode.
- One-seat: model in one seat, bots elsewhere.
- Full side: model controls landlord, or both farmers.
- Oracle landlord: bid net selects the strongest 17-card hand as landlord.
- No-shuffle: 256 deals, 3 seats = 768 games.
- No-shuffle head-to-head: same 256 deals, full side swapped, 512 games, no bots.

### Regular results

| Opponent | Overall | Landlord | Farmer |
|---|---:|---:|---:|
| Random | 0.958 | 0.973 | 0.951 |
| Greedy | 0.814 | 0.814 | 0.813 |
| Heuristic (one seat) | 0.657 | 0.768 | 0.602 |
| Heuristic (full side) | 0.770 | 0.767 | 0.772 |
| Heuristic + oracle landlord | 0.578 | 0.908 | 0.413 |

### No-shuffle results

| Model | Overall | Landlord | Farmer |
|---|---:|---:|---:|
| Regular adapter | 0.5625 | 0.4375 | 0.6250 |
| **No-shuffle expert** | **0.6549** | **0.5898** | **0.6875** |

Full-side head-to-head: regular 0.3613 vs expert **0.6387**.

### Latency

```text
first result p99 136ms
64 samples  p99 672ms
128 samples p99 875ms
512 samples p99 1792ms
under 1200ms at 128 samples: 32/32
```

“99% optimal within 1.2s” is an engineering target, not a game-theoretic proof.

## Reproduce

```powershell
python -m doudizhu.train --config configs/default.yaml
python -m doudizhu.train --config configs/noshuffle3_20261031.yaml ^
  --resume checkpoints/best_model.pt
```

```powershell
python scripts/final_eval.py --model checkpoints/best_model.pt --games 1024
python scripts/eval_oracle.py --model checkpoints/best_model.pt ^
  --bid-net checkpoints/bid_net.pt --games 512
python scripts/eval_noshuffle.py --deals 256
python scripts/head_noshuffle.py --deals 256
```

## Hardware

- RTX 5070 Ti 16GB (Blackwell, CUDA 12.8).
- ~1,300–1,400 games/s in regular training.
- GPU utilization ~10–30%; self-play dealing/rules/Numba are CPU-bound.

## Limitations

- No human evaluation; bot results are not human win rates.
- No-shuffle rule is a local approximation.
- No search distillation; it is the main next step.
- No realistic three-player bidding during training; oracle + random landlord mixture is used.
- Standard three-player DouDizhu only.
