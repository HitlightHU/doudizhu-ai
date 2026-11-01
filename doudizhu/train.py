"""自对弈 DMC 训练主循环。

用法：
  python -m doudizhu.train --config configs/default.yaml
  python -m doudizhu.train --config configs/default.yaml --resume checkpoints/latest.pt
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .actions import ACTION_FEAT_DIM
from .bidnet import load_bid_net
from .bots import HeuristicBot
from .engine import BatchEngine, HIST, HIST_DIM
from .evaluate import evaluate_team_vs_bot, evaluate_vs_bot, evaluate_vs_random
from .model import build_model, load_checkpoint, save_checkpoint
from .pool import OpponentPool
from .replay import ReplayBuffer
from .selfplay import SelfPlayRunner
from .reference import get_action_space

DEFAULTS = dict(
    seed=1,
    n_envs=256,
    games_per_iter=512,
    train_steps=32,
    batch_size=1024,
    buffer_cap=1_000_000,
    lr=3e-4,
    lr_final=1e-4,
    lr_decay_games=4_000_000,
    grad_clip=10.0,
    eps_start=0.35,
    eps_end=0.05,
    eps_decay_games=400_000,
    max_iters=100000,
    eval_every=20,
    eval_games=512,
    bot_eval_every=100,
    bot_eval_games=128,
    team_eval_every=100,
    team_eval_games=128,
    bot_prob=0.0,
    oracle_landlord=False,
    oracle_prob=1.0,
    no_shuffle=False,
    bid_net="",
    pool_size=3,
    pool_prob=0.0,
    pool_update_every=25,
    save_every=20,
    log_every=1,
    run_dir="runs/default",
    resume="",
    device="cuda",
)


def load_config(path: str | None, overrides: dict | None = None) -> dict:
    cfg = dict(DEFAULTS)
    if path:
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        cfg.update(data)
    if overrides:
        cfg.update({k: v for k, v in overrides.items() if v is not None})
    return cfg


def epsilon_at(cfg: dict, games: int) -> float:
    frac = min(1.0, games / max(1.0, cfg["eps_decay_games"]))
    return cfg["eps_start"] + (cfg["eps_end"] - cfg["eps_start"]) * frac


def make_action_features(asp):
    feat = np.zeros((asp.n + 1, ACTION_FEAT_DIM), dtype=np.float32)
    feat[:asp.n] = asp.features
    return feat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="")
    ap.add_argument("--run-dir", type=str, default="")
    ap.add_argument("--resume", type=str, default="")
    ap.add_argument("--n-envs", type=int, default=None)
    ap.add_argument("--games-per-iter", type=int, default=None)
    ap.add_argument("--train-steps", type=int, default=None)
    ap.add_argument("--max-iters", type=int, default=None)
    ap.add_argument("--eval-every", type=int, default=None)
    ap.add_argument("--save-every", type=int, default=None)
    ap.add_argument("--eval-games", type=int, default=None)
    ap.add_argument("--bot-eval-every", type=int, default=None)
    ap.add_argument("--bot-eval-games", type=int, default=None)
    ap.add_argument("--bot-prob", type=float, default=None)
    ap.add_argument("--oracle-landlord", action="store_true", default=None)
    ap.add_argument("--oracle-prob", type=float, default=None)
    ap.add_argument("--bid-net", type=str, default=None)
    ap.add_argument("--pool-prob", type=float, default=None)
    ap.add_argument("--pool-update-every", type=int, default=None)
    ap.add_argument("--team-eval-every", type=int, default=None)
    ap.add_argument("--team-eval-games", type=int, default=None)
    ap.add_argument("--buffer-cap", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config, {
        "n_envs": args.n_envs,
        "games_per_iter": args.games_per_iter,
        "train_steps": args.train_steps,
        "max_iters": args.max_iters,
        "eval_every": args.eval_every,
        "save_every": args.save_every,
        "eval_games": args.eval_games,
        "bot_eval_every": args.bot_eval_every,
        "bot_eval_games": args.bot_eval_games,
        "bot_prob": args.bot_prob,
        "oracle_landlord": args.oracle_landlord,
        "oracle_prob": args.oracle_prob,
        "bid_net": args.bid_net,
        "pool_prob": args.pool_prob,
        "pool_update_every": args.pool_update_every,
        "team_eval_every": args.team_eval_every,
        "team_eval_games": args.team_eval_games,
        "buffer_cap": args.buffer_cap,
        "lr": args.lr,
        "device": args.device,
        "run_dir": args.run_dir or None,
        "resume": args.resume or None,
    })
    run_dir = Path(cfg["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    seed = int(cfg["seed"])
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)
    if device.type == "cuda":
        print(torch.cuda.get_device_name(0), flush=True)

    asp = get_action_space()
    action_feat_np = make_action_features(asp)
    action_feat = torch.as_tensor(action_feat_np, dtype=torch.float32, device=device)
    print(f"actions={asp.n} (+pass)", flush=True)

    oracle = bool(cfg.get("oracle_landlord", False))
    bid_net = None
    if oracle:
        bid_net, bid_payload = load_bid_net(str(cfg["bid_net"]), map_location=device)
        bid_net.to(device).eval()
        print(f"bid net loaded: {cfg['bid_net']} meta={bid_payload.get('meta', {})}", flush=True)
    engine = BatchEngine(int(cfg["n_envs"]), seed=seed, auto_landlord=not oracle)
    model = build_model(asp.n + 1, cfg.get("model")).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params={n_params/1e6:.2f}M", flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg["lr"]))
    buffer = ReplayBuffer(int(cfg["buffer_cap"]), seed=seed + 1)
    mix_bot = HeuristicBot(seed=seed + 99) if float(cfg["bot_prob"]) > 0 else None
    pool = OpponentPool(model, n=int(cfg["pool_size"]), device=device) if float(cfg["pool_prob"]) > 0 else None
    runner = SelfPlayRunner(engine, model, device, action_feat_np, bot=mix_bot,
                            bot_prob=float(cfg["bot_prob"]), seed=seed + 5,
                            bid_net=bid_net, oracle_landlord=oracle,
                            oracle_prob=float(cfg.get("oracle_prob", 1.0)),
                            pool=pool, pool_prob=float(cfg["pool_prob"]),
                            no_shuffle=bool(cfg.get("no_shuffle", False)))
    print(f"opponent mixing: bot_prob={cfg['bot_prob']} oracle_landlord={oracle} "
          f"pool_prob={cfg['pool_prob']} no_shuffle={cfg.get('no_shuffle', False)}", flush=True)

    start_iter = 0
    games_done = 0
    start_games = 0
    best_win = -1.0
    best_bot = -1.0
    best_team = -1.0
    if cfg["resume"]:
        _, payload = load_checkpoint(cfg["resume"], map_location=device)
        model.load_state_dict(payload["model"], strict=False)
        if "optimizer" in payload:
            try:
                opt.load_state_dict(payload["optimizer"])
            except Exception as e:
                print("optimizer load failed:", e, flush=True)
        meta = payload.get("meta", {})
        start_iter = int(meta.get("iter", 0))
        games_done = int(meta.get("games", 0))
        best_win = float(meta.get("best_win", -1.0))
        best_bot = float(meta.get("best_bot", -1.0))
        best_team = float(meta.get("best_team", -1.0))
        print(f"resumed iter={start_iter} games={games_done}", flush=True)
    start_games = games_done
    if pool is not None:
        for pm in pool.models:
            pm.load_state_dict(model.state_dict())
            pm.eval()

    log_path = run_dir / "train.csv"
    new_log = not log_path.exists()
    log_f = open(log_path, "a", newline="", encoding="utf-8")
    log_w = csv.writer(log_f)
    if new_log:
        log_w.writerow(["iter", "games", "eps", "loss", "win_random", "ll_win", "farmer_win",
                        "win_bot", "ll_bot", "farmer_bot", "team_win", "team_ll", "team_farmer",
                        "sec", "games_per_s"])

    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(str(run_dir / "tb"))
    except Exception:
        writer = None

    def checkpoint(tag: str, it: int):
        save_checkpoint(str(run_dir / "checkpoints" / f"{tag}.pt"), model, opt,
                        {"iter": it, "games": games_done, "best_win": best_win,
                         "best_bot": best_bot, "best_team": best_team, "config": cfg})

    t_start = time.time()
    last_t = t_start
    it = start_iter
    while it < int(cfg["max_iters"]):
        eps = epsilon_at(cfg, games_done - start_games)
        model.eval()
        t0 = time.time()
        n_new = runner.run(int(cfg["games_per_iter"]), eps, buffer)
        games_done += n_new
        lr_now = cfg["lr"] + (cfg["lr_final"] - cfg["lr"]) * min(
            1.0, (games_done - start_games) / max(1.0, cfg["lr_decay_games"]))
        for pg in opt.param_groups:
            pg["lr"] = lr_now
        t_gen = time.time() - t0

        model.train()
        losses = []
        t0 = time.time()
        for _ in range(int(cfg["train_steps"])):
            if buffer.size < int(cfg["batch_size"]):
                break
            s, h, a, r = buffer.sample(int(cfg["batch_size"]))
            s_t = torch.as_tensor(s, dtype=torch.float32, device=device)
            h_t = torch.as_tensor(h, device=device).view(-1, HIST, HIST_DIM)
            a_t = torch.as_tensor(a, dtype=torch.int64, device=device)
            r_t = torch.as_tensor(r, dtype=torch.float32, device=device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
                q = model.q(s_t, h_t, a_t, action_feat[a_t])
                loss = F.mse_loss(q.float(), r_t)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["grad_clip"]))
            opt.step()
            losses.append(float(loss.detach()))
        t_train = time.time() - t0
        loss_m = float(np.mean(losses)) if losses else float("nan")
        if pool is not None and it > 0 and it % int(cfg["pool_update_every"]) == 0:
            pool.add(model)
            print(f"[pool] snapshot iter={it} updates={pool.updates}", flush=True)

        eval_out = {}
        bot_out = {}
        if it % int(cfg["eval_every"]) == 0 or it == start_iter:
            t0 = time.time()
            eval_out = evaluate_vs_random(model, action_feat, device,
                                          n_games=int(cfg["eval_games"]), seed=1234)
            if eval_out["win"] > best_win:
                best_win = eval_out["win"]
                checkpoint("best", it)
            print(f"[eval] iter={it} games={games_done} win={eval_out['win']:.3f} "
                  f"ll={eval_out['landlord_win']:.3f} farmer={eval_out['farmer_win']:.3f} "
                  f"({time.time()-t0:.1f}s)", flush=True)
            if writer:
                writer.add_scalar("eval/win_random", eval_out["win"], games_done)
                writer.add_scalar("eval/landlord_win", eval_out["landlord_win"], games_done)
                writer.add_scalar("eval/farmer_win", eval_out["farmer_win"], games_done)
        if it % int(cfg["bot_eval_every"]) == 0 or it == start_iter:
            t0 = time.time()
            bot_out = evaluate_vs_bot(model, HeuristicBot(seed=17), action_feat, device,
                                      n_games=int(cfg["bot_eval_games"]), seed=4321)
            if bot_out["win"] > best_bot:
                best_bot = bot_out["win"]
                checkpoint("best_bot", it)
            print(f"[eval-bot] iter={it} games={games_done} win={bot_out['win']:.3f} "
                  f"ll={bot_out['landlord_win']:.3f} farmer={bot_out['farmer_win']:.3f} "
                  f"({time.time()-t0:.1f}s)", flush=True)
            if writer:
                writer.add_scalar("eval/win_bot", bot_out["win"], games_done)
                writer.add_scalar("eval/ll_bot", bot_out["landlord_win"], games_done)
                writer.add_scalar("eval/farmer_bot", bot_out["farmer_win"], games_done)
        team_out = {}
        if it % int(cfg["team_eval_every"]) == 0 or it == start_iter:
            t0 = time.time()
            team_out = evaluate_team_vs_bot(model, HeuristicBot(seed=17), action_feat, device,
                                            n_games=int(cfg["team_eval_games"]), seed=555)
            if team_out["win"] > best_team:
                best_team = team_out["win"]
                checkpoint("best_team", it)
            print(f"[eval-team] iter={it} games={games_done} win={team_out['win']:.3f} "
                  f"ll={team_out['landlord_win']:.3f} farmers={team_out['farmers_win']:.3f} "
                  f"({time.time()-t0:.1f}s)", flush=True)
            if writer:
                writer.add_scalar("eval/team_win", team_out["win"], games_done)
                writer.add_scalar("eval/team_ll", team_out["landlord_win"], games_done)
                writer.add_scalar("eval/team_farmer", team_out["farmers_win"], games_done)

        now = time.time()
        sec = now - last_t
        last_t = now
        gps = n_new / max(1e-9, t_gen + t_train)
        if it % int(cfg["log_every"]) == 0:
            print(f"iter={it} games={games_done} eps={eps:.3f} loss={loss_m:.4f} "
                  f"gen={t_gen:.2f}s train={t_train:.2f}s {gps:.0f}g/s", flush=True)
        log_w.writerow([it, games_done, f"{eps:.4f}", f"{loss_m:.5f}",
                        f"{eval_out.get('win', float('nan')):.4f}",
                        f"{eval_out.get('landlord_win', float('nan')):.4f}",
                        f"{eval_out.get('farmer_win', float('nan')):.4f}",
                        f"{bot_out.get('win', float('nan')):.4f}",
                        f"{bot_out.get('landlord_win', float('nan')):.4f}",
                        f"{bot_out.get('farmer_win', float('nan')):.4f}",
                        f"{team_out.get('win', float('nan')):.4f}",
                        f"{team_out.get('landlord_win', float('nan')):.4f}",
                        f"{team_out.get('farmers_win', float('nan')):.4f}",
                        f"{sec:.2f}", f"{gps:.1f}"])
        log_f.flush()
        if writer:
            writer.add_scalar("train/loss", loss_m, games_done)
            writer.add_scalar("train/epsilon", eps, games_done)
            writer.add_scalar("train/games_per_s", gps, games_done)

        if it % int(cfg["save_every"]) == 0:
            checkpoint("latest", it)
        it += 1

    checkpoint("latest", it)
    checkpoint("final", it)
    log_f.close()
    if writer:
        writer.close()
    print("done", flush=True)


if __name__ == "__main__":
    main()
