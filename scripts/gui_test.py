import sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from doudizhu.gui import App
from doudizhu.reference import Game
from doudizhu.actions import mask_to_counts
from PIL import ImageGrab

app = App()
app.update()
print("app started")
# 人机模式
app.human_tab.new_game()
app.update()
print("HUMAN BID:", app.human_tab.bid_lbl.cget("text").replace("\n", " | "))
app.human_tab.finish_bid(True)
app.update()
print("HUMAN DOUBLE:", app.human_tab.bid_lbl.cget("text").replace("\n", " | "))
app.human_tab.start_play("不加倍")
for i in range(60):
    app.update(); time.sleep(0.05)
    g = app.human_tab.game
    if g.winner >= 0: break
    if g.turn == 0: app.human_tab.play_action(g.legal_actions()[0])
print("HUMAN winner", app.human_tab.game.winner)

# 分析模式
at = app.analysis_tab
app.nb.select(at)
app.update()
g = Game.deal(seed=42)
at.pick0.set_counts(mask_to_counts(g.hands[0]))
at.confirm_hand(); app.update()
print("ANALYSIS BID:", at.bid_lbl.cget("text").replace("\n", " | "))
at.confirm_grab(True)
at.pick2.set_counts(mask_to_counts(g.bottom))
at.confirm_bottom(); app.update()
print("ANALYSIS DOUBLE:", at.dbl_lbl.cget("text").replace("\n", " | "))
at.start_play(False)
for i in range(40):
    app.update(); time.sleep(0.1)
print("ANALYSIS state:", at.state_lbl.cget("text").replace("\n", " | ")[:250])
print("ANALYSIS win:", at.win_lbl.cget("text"))
print("ANALYSIS rec count:", len(at.rec.inner.winfo_children()))
# 让分析完成并截分析页
at.analyze()
for i in range(30):
    app.update(); time.sleep(0.1)
print("ANALYSIS win after:", at.win_lbl.cget("text"))
x = app.winfo_rootx(); y = app.winfo_rooty(); w = app.winfo_width(); h = app.winfo_height()
ImageGrab.grab(bbox=(x, y, x + w, y + h)).save(r"<repo>\checkpoints\gui_analysis.png")
print("screenshot saved")
app.destroy()
