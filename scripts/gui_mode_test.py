"""分析模式：常规/不洗牌切换、重置后保持。"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from doudizhu.gui import App


def main():
    app = App()
    app.update()
    at = app.analysis_tab
    assert at.mode_var.get() == "常规模式"
    at.mode_var.set("不洗牌模式")
    at._on_mode_change()
    app.update()
    assert app.analysis_mode == "noshuffle"
    if app.model_noshuffle is not None:
        assert app.analysis_model is app.model_noshuffle
    at._render_bid(0.75, 0.60)
    assert "建议不抢" in at.bid_lbl.cget("text"), at.bid_lbl.cget("text")
    at._render_bid(0.80, 0.60)
    assert "建议抢地主" in at.bid_lbl.cget("text"), at.bid_lbl.cget("text")
    at.reset_all()
    at.pick0.set_counts([4, 4, 4, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    at.confirm_hand()
    app.update()
    assert "不洗牌模式" in at.bid_lbl.cget("text"), at.bid_lbl.cget("text")
    at.reset_all()
    app.update()
    assert at.mode_var.get() == "不洗牌模式"
    assert app.analysis_mode == "noshuffle"
    at.mode_var.set("常规模式")
    at._on_mode_change()
    app.update()
    assert app.analysis_mode == "regular"
    assert app.analysis_model is app.model_regular
    app.destroy()
    print("PASS analysis mode switch / reset persistence")


if __name__ == "__main__":
    main()
