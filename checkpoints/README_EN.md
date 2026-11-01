# Model Card

[简体中文](README.md) | [English](README_EN.md)

Release: **v1.0.0** (2026-11-01)  
Only minimal inference state dicts are included; no optimizer. The two large weights are **GitHub Release v1.0.0 assets**; run `python scripts/download_weights.py` (range resume + SHA256 check).

## Files

| File | Purpose | Size |
|---|---|---:|
| `best_model.pt` | regular default (FP32) | 22.4 MB |
| `best_model_noshuffle.pt` | no-shuffle analysis expert (FP32) | 22.4 MB |
| `best_model_fp16.pt` | lightweight regular (FP16 storage) | 11.2 MB |
| `best_model_noshuffle_fp16.pt` | lightweight no-shuffle (FP16 storage) | 11.2 MB |
| `bid_net.pt` | regular-mode bid prior | 84 KB |

SHA256:

```text
best_model.pt          258E793FB84C09F92246D8696B5DEBBF5CDA87C7C445ADDED995344318123E49
best_model_noshuffle.pt BD91D6924EABC5DCC3A11C2AF75EB69113E4217BFFBB5E636DA2A11EA7A6AE34
best_model_fp16.pt     E8E98EF03EEC7A72428133263F9B400D397992E47814D94C55D623425BBEBC01
best_model_noshuffle_fp16.pt 88F4A09586DE85B6661625D5FB170F16D156DF68752DE8D82FC3261E2749E13B
bid_net.pt             64B8CA85A625D5F094B01F61874D03059915E6D34775EAF9E8A6CFFB226EFC6F
```

## Regular model

- 5.61M parameters: 4.56M v2 + 1.05M residual adapter, 3-checkpoint soup.
- 26,064 + PASS actions.
- 1024 fixed deals:

| Opponent | Overall | Landlord | Farmer |
|---|---:|---:|---:|
| Random | 0.958 | 0.973 | 0.951 |
| Greedy | 0.814 | 0.814 | 0.813 |
| Heuristic (one seat) | 0.657 | 0.768 | 0.602 |
| Heuristic (full side) | 0.770 | 0.767 | 0.772 |
| Heuristic + oracle landlord | 0.578 | 0.908 | 0.413 |

## No-shuffle expert

- Continued from the regular 5.61M for 80 iterations / 1,313,483 games.
- 256 no-shuffle deals / 768 games:

| Model | Overall | Landlord | Farmer |
|---|---:|---:|---:|
| Regular adapter | 0.5625 | 0.4375 | 0.6250 |
| **Expert** | **0.6549** | **0.5898** | **0.6875** |

Full-side head-to-head over 512 games: regular 0.3613 vs expert **0.6387**.

No-shuffle mode uses the expert's own parameters for initial bid, live bid, top-3, both-camp win rates and doubling. It does **not** use `bid_net.pt`. Bid rule: **p_landlord > 0.75**.

## Usage

The GUI loads these by default. Environment overrides:

```powershell
$env:DOUDIZHU_MODEL = "checkpoints/best_model.pt"
$env:DOUDIZHU_MODEL_NOSHUFFLE = "checkpoints/best_model_noshuffle.pt"
$env:DOUDIZHU_BID = "checkpoints/bid_net.pt"
```

## Metrics files

`model_info.json`, `final_eval.json`, `noshuffle_eval.json`, `head_noshuffle.json`, `latency.json`, `eval_oracle.txt`.

## Limitations

No human evaluation; all numbers are against bots/models.  
No-shuffle is a reproducible local approximation, not an official server reimplementation.
