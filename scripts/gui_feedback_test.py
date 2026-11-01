"""针对本轮 7 条反馈的 GUI 测试：速度/固定栏/记牌器/PASS/非推荐/实时胜率/双方胜率。"""
import sys, time, tkinter as tk
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from doudizhu.gui import App
from doudizhu.reference import Game, counts_to_mask


def analysis_ready(app):
    at = app.analysis_tab
    app.nb.select(at)
    at.reset_all()
    hand = [4, 4, 4, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    at.pick0.set_counts(hand)
    at.confirm_hand()
    at.confirm_grab(True)
    at.pick2.set_counts([0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    at.confirm_bottom()
    at.start_play(False)
    return at


def pump(app, seconds):
    t = time.time()
    while time.time() - t < seconds:
        app.update()
        time.sleep(0.02)


def pump_until(app, pred, timeout=4.0):
    t = time.time()
    while time.time() - t < timeout:
        app.update()
        if pred():
            return True
        time.sleep(0.03)
    return bool(pred())


def test_counter_turn_live():
    app = App(); app.withdraw()
    at = analysis_ready(app)
    s = at.session
    assert s.turn == s.human_seat
    assert "该你出牌" in str(at.turn_badge.cget("text"))
    unseen = s.unseen_counts()
    for r, lab in enumerate(at.counter_labels):
        assert str(lab.cget("text")) == str(int(unseen[r]))
    ll, ff = s.camp_value(app.model, app.device, app.action_feat, samples=8)
    assert abs(ll + ff - 1.0) < 1e-6
    pump(app, 0.6)
    txt = str(at.camp_lbl.cget("text"))
    assert "地主" in txt and "农民" in txt and "%" in txt
    print("PASS counter / turn badge / both-camp live estimate:", txt)
    app.destroy()


def test_bottom_pass_and_manual():
    app = App(); app.withdraw()
    at = analysis_ready(app)
    s = at.session
    # 人类先出一手，对手1出一手，对手2过，回到人类跟牌
    s.apply(0, s.legal_actions()[0]); at._refresh_state()
    s.apply(s.turn, s.legal_actions()[0]); at._refresh_state()
    s.apply(s.turn, s.asp.n); at._refresh_state()
    assert s.turn == s.human_seat
    assert "disabled" not in at.btn_pass4.state()
    at.btn_pass4.invoke()
    assert s.history[-1][1] == -1 and s.last_act == -1
    assert "过" in str(at.msg.cget("text"))
    print("PASS bottom PASS button")
    app.destroy()

    # 非推荐动作也能手动出（新开一局，人类自由出牌）
    app = App(); app.withdraw()
    at = analysis_ready(app)
    at.analyze()
    assert pump_until(app, lambda: len(at.last_best_actions) >= 3, 3.0), "等待实时前三选超时"
    s = at.session
    legal = [a for a in s.legal_actions() if a != s.asp.n]
    not_best = [a for a in legal if a not in at.last_best_actions]
    assert not_best, "所有合法动作都在前三选？"
    a = not_best[0]
    before = len(at.history)
    at.apply_human(a)
    assert len(at.history) == before + 1
    print("PASS manual non-recommended action")
    app.destroy()


def test_sort_and_relative_turn():
    app = App(); app.withdraw()
    at = app.analysis_tab
    app.nb.select(at)
    at.reset_all()
    at.pick0.set_counts([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    ranks = [r for r, _ in at.pick0.chip_buttons]
    assert ranks == sorted(ranks, reverse=True), f"初始手牌应大到小: {ranks}"
    at.pick4.set_counts([0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    r4 = [r for r, _ in at.pick4.chip_buttons]
    assert r4 == sorted(r4), f"出牌选择应保持小到大: {r4}"
    at.pick4.clear()
    at.pick2.set_counts([0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    r2 = [r for r, _ in at.pick2.chip_buttons]
    assert r2 == sorted(r2, reverse=True), f"底牌选择应大到小: {r2}"
    at.pick2.clear()
    # 构造一个人类地主的牌局，测试上/下家文案
    at.reset_all()
    at.pick0.set_counts([4, 4, 4, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    at.confirm_hand(); at.confirm_grab(True)
    at.pick2.set_counts([0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    at.confirm_bottom(); at.start_play(False)
    s = at.session
    s.turn = 1; at._refresh_state()
    assert "下家" in str(at.turn_badge.cget("text"))
    s.turn = 2; at._refresh_state()
    assert "上家" in str(at.turn_badge.cget("text"))
    s.turn = 0; at._refresh_state()
    assert "该你出牌" in str(at.turn_badge.cget("text"))
    app.destroy()
    print("PASS descending initial sort / ascending action sort / relative turn text")


def test_human_turn_badge_live():
    app = App(); app.withdraw()
    ht = app.human_tab
    app.nb.select(ht)
    ht.new_game(); ht.finish_bid(True); ht.start_play("不加倍")
    pump(app, 0.6)
    assert "该你出牌" in str(ht.turn_badge.cget("text"))
    txt = str(ht.camp_lbl.cget("text"))
    assert "地主" in txt and "农民" in txt and "%" in txt
    s = ht.make_session()
    ll, ff = s.camp_value(app.model, app.device, app.action_feat, samples=8)
    assert abs(ll + ff - 1.0) < 1e-6
    print("PASS human turn badge / both-camp live estimate:", txt)
    app.destroy()


test_counter_turn_live()
test_bottom_pass_and_manual()
test_sort_and_relative_turn()
test_human_turn_badge_live()
print("ALL FEEDBACK TESTS PASSED")
