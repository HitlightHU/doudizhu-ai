"""第二批 GUI 测试：出牌合法性、PASS、ribbon 历史/撤销/重置/结束。"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import tkinter as tk

from doudizhu.actions import SINGLE, PAIR, counts_to_mask, mask_to_counts
from doudizhu.analysis import AnalysisSession
from doudizhu.gui import App
from doudizhu.reference import Game, get_action_space


def test_rules_legality():
    asp = get_action_space()
    a3 = next(i for i in range(asp.n) if int(asp.typ[i]) == SINGLE and int(asp.rank[i]) == 0)
    s4 = next(i for i in range(asp.n) if int(asp.typ[i]) == SINGLE and int(asp.rank[i]) == 1)
    p4 = next(i for i in range(asp.n) if int(asp.typ[i]) == PAIR and int(asp.rank[i]) == 1)
    h0 = [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]      # 3,9
    h1 = [0, 2, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]      # 4,4,8
    h2 = [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]      # 5,7
    hands = [counts_to_mask(h0), counts_to_mask(h1), counts_to_mask(h2)]
    g = Game.from_state(hands=hands, bottom=0, landlord=0, turn=0)
    assert not g.is_legal(asp.n), "leading PASS must be illegal"
    g.step(a3)
    assert g.turn == 1 and g.last_act == a3
    assert g.is_legal(asp.n), "following PASS must be legal"
    assert g.is_legal(s4), "higher single must be legal"
    assert not g.is_legal(p4), "pair cannot beat a single"
    assert asp.n in g.legal_actions()
    print("PASS rule engine legality / PASS")


def test_analysis_legality():
    asp = get_action_space()
    g = Game.deal(seed=11)
    s = AnalysisSession.start(g.hands[0], True, g.bottom, 0)
    assert not s.is_legal_for(0, asp.n), "leading PASS must be illegal"
    try:
        s.apply(0, asp.n)
        raise AssertionError("apply leading pass should raise")
    except ValueError:
        pass
    a = s.legal_actions()[0]
    s.apply(0, a)
    assert s.is_legal_for(s.turn, asp.n), "next player may pass"
    assert asp.n in s.legal_actions(), "PASS must be in legal actions"
    # 对手手牌只剩 1 张时不能记一对
    s2 = AnalysisSession.start(g.hands[0], True, g.bottom, 0)
    s2.hand_sizes[1] = 1
    pair = next(i for i in range(asp.n) if int(asp.typ[i]) == PAIR)
    assert not s2.is_legal_for(1, pair)
    print("PASS analysis legality / PASS")


def _analysis_ready(app):
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


def test_analysis_history():
    app = App()
    app.withdraw()
    at = _analysis_ready(app)
    assert len(at.history) == 1 and at.hcursor == 0
    a = at.session.legal_actions()[0]
    at.apply_human(a)
    assert len(at.history) == 2 and at.hcursor == 1
    after = at.session.human_hand
    at.prev_turn()
    assert at.hcursor == 0
    at.next_turn()
    assert at.hcursor == 1
    assert at.session.human_hand == after
    at.undo()  # pick4 空 -> 回退一步
    assert at.hcursor == 0
    at.pick4.add(0)
    at.undo()  # 有选牌 -> 只清空选牌
    assert at.pick4.total() == 0 and at.hcursor == 0
    at.reset_play()
    assert at.hcursor == 0 and not at.ended
    at.end_game()
    assert at.ended
    at.reset_play()
    assert not at.ended
    app.destroy()
    print("PASS analysis history prev/next/undo/reset/end")


def test_lead_pass_buttons():
    app = App()
    app.withdraw()
    asp = get_action_space()
    a3 = next(i for i in range(asp.n) if int(asp.typ[i]) == SINGLE and int(asp.rank[i]) == 0)
    # 分析模式：我出一手，两家 PASS 后我重新自由出牌
    at = _analysis_ready(app)
    s = at.session
    a = s.legal_actions()[0]
    s.apply(0, a)
    s.apply(s.turn, asp.n)
    s.apply(s.turn, asp.n)
    at._refresh_state()
    assert s.turn == 0 and s.last_act == -1
    assert "disabled" in at.btn_pass4.state(), "分析模式自由出牌时过按钮必须禁用"
    # 人机模式：构造自由出牌状态
    ht = app.human_tab
    h0 = [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    h1 = [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    h2 = [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    ht.game = Game.from_state(hands=[counts_to_mask(h0), counts_to_mask(h1), counts_to_mask(h2)],
                              bottom=0, landlord=0, turn=0, last_act=-1)
    ht.phase = "play"
    ht.ended = False
    ht.refresh()
    assert "disabled" in ht.btn_pass.state(), "人机自由出牌时过按钮必须禁用"
    ht.game.last_act = a3
    ht.game.last_player = 2
    ht.refresh()
    assert "disabled" not in ht.btn_pass.state(), "跟牌时过按钮应可用"
    app.destroy()
    print("PASS lead/follow pass buttons")


def test_back_step():
    app = App()
    app.withdraw()
    at = app.analysis_tab
    app.nb.select(at)
    at.reset_all()
    assert "disabled" in at.btn_back.state()
    at.pick0.set_counts([4, 4, 4, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    at.confirm_hand()
    assert at.step == 1
    assert "disabled" not in at.btn_back.state()
    at.back_step()
    assert at.step == 0 and at.pick0.total() == 17
    assert "disabled" in at.btn_back.state()
    at.confirm_hand()
    at.confirm_grab(True)
    assert at.step == 2
    bottom = [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    at.pick2.set_counts(bottom)
    at.back_step()
    assert at.step == 1
    at.confirm_grab(True)
    assert at.step == 2 and at.pick2.total() == 3, "返回叫地主再回来，底牌应保留"
    at.confirm_bottom()
    assert at.step == 3
    at.back_step()
    assert at.step == 2 and at.pick2.total() == 3
    at.confirm_bottom()
    assert at.step == 3
    at.start_play(False)
    assert at.step == 4 and len(at.history) == 1
    assert "disabled" not in at.btn_back.state(), "未出牌时应可回加倍"
    at.back_step()
    assert at.step == 3
    at.start_play(False)
    a = at.session.legal_actions()[0]
    at.apply_human(a)
    assert len(at.history) == 2
    assert "disabled" in at.btn_back.state(), "已出牌后设置阶段应禁止回退"
    at.reset_all()
    app.destroy()
    print("PASS analysis back-step state preservation")


def test_human_history():
    app = App()
    app.withdraw()
    ht = app.human_tab
    app.nb.select(ht)
    ht.new_game()
    ht.finish_bid(True)
    ht.start_play("不加倍")
    assert len(ht.history) == 1 and ht.hcursor == 0
    a = ht.game.legal_actions()[0]
    ht.play_action(a)
    assert len(ht.history) == 2 and ht.hcursor == 1
    ht.prev_turn()
    assert ht.hcursor == 0
    ht.next_turn()
    assert ht.hcursor == 1
    ht.end_game()
    assert ht.ended
    ht.new_game()
    assert not ht.ended and len(ht.history) == 0
    app.destroy()
    print("PASS human history prev/next/end/reset")


test_rules_legality()
test_analysis_legality()
test_analysis_history()
test_lead_pass_buttons()
test_back_step()
test_human_history()
print("ALL GUI ROBUSTNESS 2 TESTS PASSED")
