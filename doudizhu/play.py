"""命令行人机对局。

示例：
  python -m doudizhu.play --model checkpoints/best.pt --seat 0
  python -m doudizhu.play --model checkpoints/best.pt --seat 1 --opponent heuristic
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from .actions import TYPE_NAMES
from .bots import make_bot
from .engine import BatchEngine, HIST, HIST_DIM
from .model import load_checkpoint
from .reference import Game, action_str, get_action_space, mask_str
from .selfplay import select_actions_batch
from .train import make_action_features


def sync_engine(engine: BatchEngine, g: Game):
    engine.hands[0, 0] = g.hands[0]
    engine.hands[0, 1] = g.hands[1]
    engine.hands[0, 2] = g.hands[2]
    engine.played[0] = g.played
    engine.bottom[0] = g.bottom
    engine.landlord[0] = g.landlord
    engine.turn[0] = g.turn
    engine.last_act[0] = g.last_act
    engine.last_player[0] = g.last_player
    engine.passes[0] = g.passes
    engine.winner[0] = g.winner
    hist = g.history[-HIST:]
    engine.hist_len[0] = len(hist)
    for i in range(HIST):
        engine.hist_a[0, i] = -1
        engine.hist_p[0, i] = -1
    for i, (p, a) in enumerate(hist):
        engine.hist_a[0, i] = -1 if a == g.pass_id else int(a)
        engine.hist_p[0, i] = int(p)


def model_action(model, engine, action_feat, device):
    st, hist = engine.observe()
    idx, cnt = engine.legal()
    a = int(select_actions_batch(model, device, action_feat, st, hist, idx, cnt, 0.0)[0])
    return a


def describe(asp, a):
    if a == asp.n:
        return "过"
    t = int(asp.typ[a])
    return f"{TYPE_NAMES[t]}  {mask_str(int(asp.mask[a]))}"


def show_table(g, asp):
    print("-" * 62)
    for p in range(3):
        role = "地主" if p == g.landlord else "农民"
        mark = " <-- 该你" if p == g.turn else ""
        print(f"玩家{p} {role} 剩{g.card_count(p):2d}张{mark}")
    print(f"底牌: {mask_str(g.bottom)}")
    if g.last_act >= 0:
        print(f"上一手: 玩家{g.last_player} -> {describe(asp, g.last_act)}")
    else:
        print("上一手: 无（自由出牌）")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="checkpoints/best.pt")
    ap.add_argument("--seat", type=int, default=0, help="人类座位 0/1/2")
    ap.add_argument("--opponent", type=str, default="model",
                    choices=["model", "heuristic", "greedy", "random"])
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--auto", action="store_true", help="人类座位也由 bot 控制，用于自动演示")
    args = ap.parse_args()

    asp = get_action_space()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = None
    action_feat = None
    if args.opponent == "model" or True:
        try:
            model, payload = load_checkpoint(args.model, map_location=device)
            model.to(device).eval()
            action_feat = torch.as_tensor(make_action_features(asp), dtype=torch.float32, device=device)
            print(f"模型已加载: {args.model}")
        except Exception as e:
            print(f"模型加载失败（{e}），改用启发式机器人")
            model = None
    bots = {}
    for p in range(3):
        if p == args.seat:
            continue
        if args.opponent == "model" and model is not None:
            bots[p] = None
        else:
            bots[p] = make_bot(args.opponent if args.opponent != "model" else "heuristic", seed=(args.seed or 0) + p)

    auto_bot = make_bot("heuristic", seed=(args.seed or 0) + 99) if args.auto else None
    engine = BatchEngine(1, seed=args.seed or 1)
    g = Game(seed=args.seed)
    print(f"\n你是玩家{args.seat}，身份：{'地主' if g.landlord == args.seat else '农民'}")
    print(f"你的手牌: {mask_str(g.hands[args.seat])}")
    print(f"底牌: {mask_str(g.bottom)}")

    while g.winner < 0:
        p = g.turn
        show_table(g, asp)
        if p == args.seat and auto_bot is not None:
            a = auto_bot.choose(g.hands[p], g.last_act, g.legal_actions(), p,
                                g.landlord, g.last_player,
                                [g.card_count(x) for x in range(3)])
            print(f"（自动）你出: {describe(asp, a)}")
        elif p == args.seat:
            legal = g.legal_actions()
            print("可选出牌：")
            for i, a in enumerate(legal):
                print(f"  [{i:2d}] {describe(asp, a)}")
            while True:
                cmd = input("选择编号 (p=过, q=退出): ").strip()
                if cmd.lower() == "q":
                    return
                if cmd.lower() in ("p", "pass"):
                    if g.pass_id in legal:
                        a = g.pass_id
                        break
                    print("现在不能过。")
                    continue
                try:
                    a = legal[int(cmd)]
                    break
                except Exception:
                    print("输入无效。")
        else:
            if bots.get(p) is not None:
                a = bots[p].choose(g.hands[p], g.last_act, g.legal_actions(), p,
                                   g.landlord, g.last_player,
                                   [g.card_count(x) for x in range(3)])
            else:
                sync_engine(engine, g)
                a = model_action(model, engine, action_feat, device)
                if a not in g.legal_actions():
                    a = g.legal_actions()[0]
            print(f"玩家{p} 出: {describe(asp, a)}")
        g.step(a)
        if p == args.seat and g.hands[p]:
            print(f"你的手牌: {mask_str(g.hands[p])}")

    print("\n" + "=" * 62)
    if args.seat == g.landlord:
        human_win = g.winner == args.seat
    else:
        human_win = g.winner != g.landlord
    print(f"玩家{g.winner} 获胜！你是{'赢' if human_win else '输'}")
    print(f"地主: 玩家{g.landlord}；你的角色: {'地主' if args.seat == g.landlord else '农民'}")
    for p in range(3):
        print(f"玩家{p} 剩余: {mask_str(g.hands[p])}")


if __name__ == "__main__":
    main()
