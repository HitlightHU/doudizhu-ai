import sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from doudizhu.gui import App

app = App()
app.withdraw()
ht = app.human_tab
ht.new_game()
app.update()
ht.finish_bid(True)
app.update()
ht.start_play("不加倍")
for i in range(200):
    app.update(); time.sleep(0.03)
    g = ht.game
    if g.winner >= 0:
        break
    if g.turn == 0:
        ht.show_recommend()
        for _ in range(80):
            app.update(); time.sleep(0.03)
            if ht.rec_active:
                break
        print("STATUS:", ht.status.cget("text"))
        print("WIN:", ht.win_lbl.cget("text"))
        print("BUTTONS:", len(ht.actions.inner.winfo_children()))
        for w in ht.actions.inner.winfo_children()[:3]:
            try: print("  ", w.cget("text"))
            except Exception: pass
        break
print("phase", ht.phase, "turn", ht.game.turn)
app.destroy()
