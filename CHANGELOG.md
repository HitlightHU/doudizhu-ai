# Changelog

## v1.0.0 — 2026-11-01

### Models

- Release 5.61M regular adapter soup (`checkpoints/best_model.pt`).
- Release dedicated 5.61M no-shuffle expert (`checkpoints/best_model_noshuffle.pt`).
- Release regular bid prior (`checkpoints/bid_net.pt`).
- Remove optimizer/backup checkpoints from the release package.

### GUI

- Analysis mode: regular / no-shuffle selector; mode persists across reset.
- No-shuffle mode uses its own model and rollout-based win-rate estimates.
- No-shuffle bid rule: only `p_landlord > 0.75` recommends grabbing.
- Initial hand/bottom card library sorted high-to-low; play picker remains low-to-high.
- Anytime search updates top-3 and both-camp win rates roughly every 0.2s.
- Large turn badge: “该你出牌！！！” / “该你的上家出牌” / “该你的下家出牌”.

### Documentation

- Chinese-first README with English switch.
- GUI guide, training report and model card, each with an English counterpart.
- AI disclosure, acknowledgements, MIT license and citation metadata.

### Evaluation

- Regular: random 0.958 / greedy 0.814 / heuristic one-seat 0.657 / team 0.770 / oracle 0.578.
- No-shuffle: regular 0.5625 vs expert 0.6549 (768 games).
- No-shuffle full-side head-to-head: regular 0.3613 vs expert 0.6387 (512 games).
