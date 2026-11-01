"""GUI 边界/鲁棒性测试：选牌上限、剩余张数、手牌点选放回、reset、底牌耗尽。"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import time
import numpy as np
import tkinter as tk

from doudizhu.gui import App, CardPicker, RANK_NAMES, TOTAL_COUNTS
from doudizhu.reference import Game, mask_to_counts
from doudizhu.analysis import AnalysisSession


def test_picker_basic():
    root = tk.Tk()
    root.withdraw()
    p = CardPicker(root, max_total=17)
    for _ in range(10):
        p.add(0)
    assert int(p.counts[0]) == 4, p.counts
    assert p.buttons[0].cget("state") == "disabled"
    assert p.rem_labels[0].cget("text") == "剩 0"
    for r in range(1, 5):
        for _ in range(4):
            p.add(r)
    assert p.total() == 17
    p.add(5)
    assert p.total() == 17, "must not exceed 17"
    chips = [w for w in p.hand_strip.winfo_children() if isinstance(w, tk.Button)]
    assert len(chips) == 17
    p.remove_at(0)
    assert p.total() == 16
    assert int(p.counts[0]) == 3
    p.add(0)
    assert p.total() == 17
    p.undo()
    assert p.total() == 16
    p.clear()
    assert p.total() == 0
    # 每个点数上限
    p.set_limits(np.array([0] * 15, dtype=np.int8), 3)
    p.add(0)
    assert p.total() == 0
    p.set_limits(np.array([2, 1] + [0] * 13, dtype=np.int8), 3)
    p.add(0); p.add(0); p.add(0)
    assert int(p.counts[0]) == 2
    p.add(1); p.add(2)
    assert p.total() == 3
    p.set_counts([4, 4, 4, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    assert p.total() <= 3
    root.destroy()
    print("PASS picker basic / max 4 / max total / chips / remove")


def test_picker_fixed_height_and_speed():
    root = tk.Tk()
    root.withdraw()
    p = CardPicker(root, max_total=17)
    p.pack()
    p.update_idletasks()
    h0 = p.hand_strip.winfo_height()
    for r in range(4):
        for _ in range(4):
            p.add(r)
    p.add(4)
    assert p.total() == 17 and len(p.chip_buttons) == 17
    p.update_idletasks()
    assert p.hand_strip.winfo_height() == h0, "手牌栏必须是固定高度"
    t0 = time.perf_counter()
    for _ in range(100):
        p.remove_rank(0)
        p.add(0)
        p.update_idletasks()
    dt = time.perf_counter() - t0
    assert p.total() == 17 and len(p.chip_buttons) == 17
    assert dt < 2.0, f"rapid add too slow: {dt:.2f}s"
    print(f"PASS fixed hand strip / 100 add-remove cycles {dt*1000:.0f} ms")
    root.destroy()


def test_bottom_and_reset():
    app = App()
    app.withdraw()
    at = app.analysis_tab
    # 4×3,4,5,6 + 7 = 17，剩牌中 3/4/5/6 均为 0
    hand_counts = [4, 4, 4, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    at.pick0.set_counts(hand_counts)
    assert at.pick0.total() == 17
    at.confirm_hand()
    assert at.step == 1
    assert np.array_equal(at.pick2.limits, at.remain_after_hand)
    for r in range(4):
        assert int(at.remain_after_hand[r]) == 0
        assert at.pick2.buttons[r].cget("state") == "disabled"
        before = at.pick2.total()
        at.pick2.add(r)
        assert at.pick2.total() == before
    at.confirm_grab(True)
    assert at.step == 2
    # 底牌只能从剩余牌里选：7,8,9
    bottom_counts = [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    at.pick2.set_counts(bottom_counts)
    assert at.pick2.total() == 3
    at.confirm_bottom()
    assert at.session is not None
    assert at.session.hand_sizes[0] == 20
    at.start_play(False)
    assert at.step == 4
    assert at.pick4.max_total == at.session.hand_sizes[0]
    assert np.array_equal(at.pick4.limits, mask_to_counts(at.session.human_hand))
    at.reset_all()
    assert at.step == 0
    assert at.pick0.total() == 0 and at.pick2.total() == 0 and at.pick4.total() == 0
    assert at.session is None
    app.destroy()
    print("PASS bottom exhausted / reset / play picker limits")


def test_session_edges():
    g = Game.deal(seed=7)
    s = AnalysisSession.start(g.hands[0], True, g.bottom, 0)
    try:
        s.apply(0, s.asp.n)
        raise AssertionError("leading pass must be illegal")
    except ValueError:
        pass
    # 出牌后手牌/张数减少，且不能选超过手牌
    legal = s.legal_actions()
    assert legal
    a = legal[0]
    before = int(mask_to_counts(s.human_hand).sum())
    s.apply(0, a)
    assert int(mask_to_counts(s.human_hand).sum()) == before - int(s.asp.ncards[a])
    assert s.hand_sizes[0] == before - int(s.asp.ncards[a])
    print("PASS session boundary")


test_picker_basic()
test_picker_fixed_height_and_speed()
test_bottom_and_reset()
test_session_edges()
print("ALL GUI ROBUSTNESS TESTS PASSED")
