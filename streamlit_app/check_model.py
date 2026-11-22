import sys
import time

if len(sys.argv) != 2:
    sys.exit("الاستخدام: py check_model.py path/to/audio.wav")

from nabra_explain import explain, load_model, predict

t = time.time(); load_model(); print(f"✓ الموديل تحمّل ({time.time() - t:.0f} ثانية)")

t = time.time(); r = predict(sys.argv[1])
print(f"✓ predict ({time.time() - t:.1f} ثانية): {r}")
assert r["label"] in ("real", "fake") and 0 <= r["confidence"] <= 1

t = time.time(); e = explain(sys.argv[1])
print(f"✓ explain ({time.time() - t:.1f} ثانية): segments = {e['segments']}")
print("كل شي تمام، شغلي الواجهة: py -m streamlit run app.py")
