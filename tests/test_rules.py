"""规则与动作空间测试（不依赖 pytest，可直接运行）。"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np

from doudizhu.actions import (
    PLANE, PLANE1, PLANE2, STRAIGHT, PAIR_STRAIGHT, BOMB, ROCKET, FOUR2, FOUR2PAIR,
    TRIO1, TRIO2, counts_to_mask, mask_to_counts, TYPE_NAMES,
)
from doudizhu.reference import Game, get_action_space
from doudizhu.analysis import should_grab, should_grab_noshuffle

asp = get_action_space()
RANK = {"3": 0, "4": 1, "5": 2, "6": 3, "7": 4, "8": 5, "9": 6,
        "T": 7, "10": 7, "J": 8, "Q": 9, "K": 10, "A": 11, "2": 12, "x": 13, "X": 14}


def cards(s: str) -> int:
    cnt = [0] * 15
    i = 0
    s = s.replace(" ", "")
    while i < len(s):
        if s[i] in ("1",) and i + 1 < len(s) and s[i + 1] == "0":
            cnt[RANK["10"]] += 1
            i += 2
            continue
        cnt[RANK[s[i]]] += 1
        i += 1
    return counts_to_mask(cnt)


def find_action(mask):
    idx = np.nonzero(asp.mask == np.int64(mask))[0]
    return int(idx[0]) if len(idx) else None


def kinds(mask):
    a = find_action(mask)
    assert a is not None, f"动作不在表中: {mask}"
    return int(asp.typ[a]), int(asp.rank[a]), int(asp.chain[a])


def test_action_table():
    assert asp.n > 20000
    assert len(set(int(x) for x in asp.mask)) == asp.n
    assert np.all(asp.counts.sum(1) == asp.ncards)
    assert np.all(asp.counts <= 4)
    # 所有 mask 都是规范形式（最低花色位）
    for i in range(0, asp.n, 997):
        m = int(asp.mask[i])
        assert counts_to_mask(mask_to_counts(m)) == m


def test_patterns():
    assert kinds(cards("333444555")) == (PLANE, 0, 3)
    assert kinds(cards("33344455")) == (PLANE1, 0, 2)
    assert kinds(cards("3334445566")) == (PLANE2, 0, 2)
    # 用户给的飞机带翅膀例子：n 个三顺 + n 个单牌 / n 个对子
    assert kinds(cards("3334449Q")) == (PLANE1, 0, 2)
    assert kinds(cards("555666777JJQQKK")) == (PLANE2, 2, 3)
    assert kinds(cards("333444JJQQ")) == (PLANE2, 0, 2)
    assert kinds(cards("5556667779JQ")) == (PLANE1, 2, 3)
    assert kinds(cards("33344499")) == (PLANE1, 0, 2)  # 一对可拆成两张单翅
    assert find_action(cards("3334449")) is None       # 单翅少一张
    assert find_action(cards("3334449QK")) is None     # 单翅多一张
    assert find_action(cards("333444999")) is None     # 三张同点不当单翅，避免纯飞机歧义
    assert kinds(cards("34567")) == (STRAIGHT, 0, 5)
    assert kinds(cards("TJQKA")) == (STRAIGHT, 7, 5)
    assert kinds(cards("334455")) == (PAIR_STRAIGHT, 0, 3)
    assert kinds(cards("3333")) == (BOMB, 0, 0)
    assert kinds(cards("333355")) == (FOUR2, 0, 0)
    assert kinds(cards("33335566")) == (FOUR2PAIR, 0, 0)
    assert kinds(cards("xX")) == (ROCKET, 13, 0)
    assert find_action(cards("3334445")) is None       # 不完整飞机
    assert find_action(cards("JQKA2")) is None         # 顺子不能含 2
    assert find_action(cards("KA2")) is None
    assert find_action(cards("3344")) is None          # 连对至少 3 对
    assert find_action(cards("333444555666")) == find_action(cards("333444555666"))
    # 333444555 只能是三顺，不能同时是飞机带单
    a = find_action(cards("333444555"))
    assert int(asp.typ[a]) == PLANE


def test_legal_follow():
    # 顺子 34567，手牌 45678 / 炸弹 / 王炸
    hand = cards("45678") | cards("3333") | cards("xX")
    g = Game.from_state(hands=[hand, cards("A"), cards("A")], bottom=0, landlord=0, turn=0,
                        last_act=find_action(cards("34567")))
    legal = set(g.legal_actions())
    assert find_action(cards("45678")) in legal
    assert find_action(cards("34567")) not in legal
    assert find_action(cards("3333")) in legal
    assert find_action(cards("xX")) in legal
    assert g.pass_id in legal
    # 对子不能压顺子
    assert find_action(cards("55")) not in legal

    # 王炸之后只能过
    g = Game.from_state(hands=[cards("45678"), cards("A"), cards("A")], bottom=0, landlord=0,
                        turn=0, last_act=find_action(cards("xX")))
    assert g.legal_actions() == [g.pass_id]

    # 炸弹 3333 之后：更大的炸弹/王炸/过
    g = Game.from_state(hands=[cards("4444xX5"), cards("A"), cards("A")], bottom=0, landlord=0,
                        turn=0, last_act=find_action(cards("3333")))
    legal = set(g.legal_actions())
    assert find_action(cards("4444")) in legal
    assert find_action(cards("xX")) in legal
    assert find_action(cards("3333")) not in legal

    # 单张 3 之后，对子不能压
    g = Game.from_state(hands=[cards("55"), cards("A"), cards("A")], bottom=0, landlord=0,
                        turn=0, last_act=find_action(cards("3")))
    legal = set(g.legal_actions())
    assert find_action(cards("55")) not in legal
    assert find_action(cards("5")) in legal


def test_pass_rules():
    g = Game.from_state(hands=[cards("34567"), cards("A"), cards("A")], bottom=0, landlord=0, turn=0)
    assert g.pass_id not in g.legal_actions()
    g.step(find_action(cards("34567")))
    # 下家可过
    g2 = Game.from_state(hands=[cards("A"), cards("34567"), cards("A")], bottom=0, landlord=1, turn=0,
                         last_act=find_action(cards("34567")))
    assert g2.is_legal(g2.pass_id)


def test_bid_rule():
    assert should_grab(0.65, 0.60)      # 地主胜率更高
    assert should_grab(0.71, 0.80)      # 地主胜率 >70%
    assert should_grab(0.60, 0.64)      # 低于农民不超过 5%
    assert not should_grab(0.60, 0.70)  # 低 10% 且不 >70%
    assert should_grab(0.50, 0.51)      # 低 1% 可抢
    assert not should_grab_noshuffle(0.75)  # 不洗牌必须 >75%
    assert should_grab_noshuffle(0.7501)
    assert not should_grab_noshuffle(0.70)


def test_four_with_two():
    # 3333 + 55 可以；3333 + 5 + 6 可以；3333 + 555 不行（无此牌型）
    for s in ("333355", "333356", "3333xX"):
        assert find_action(cards(s)) is not None, s
    assert find_action(cards("33335566")) is not None
    # 四带两对不能带同一对
    assert find_action(cards("33335555")) is None


def test_end_to_end_random():
    rng = np.random.default_rng(9)
    for _ in range(30):
        g = Game(seed=int(rng.integers(1 << 30)))
        while g.winner < 0:
            legal = g.legal_actions()
            assert legal
            a = int(rng.choice(legal))
            assert g.is_legal(a)
            g.step(a)
        assert g.card_count(g.winner) == 0


if __name__ == "__main__":
    test_action_table()
    test_patterns()
    test_legal_follow()
    test_pass_rules()
    test_four_with_two()
    test_end_to_end_random()
    print("all rule tests passed")
