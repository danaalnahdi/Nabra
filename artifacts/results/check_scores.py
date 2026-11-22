import os, sys
import numpy as np, pandas as pd
from sklearn.metrics import roc_curve

M = r"C:\Users\Aisha\OneDrive\سطح المكتب\vss\Grad\artifacts\aradf_eda\manifests"
run = sys.argv[1]

def eer(d):
    f, t, _ = roc_curve(d.label, d.score)
    fn = 1 - t
    i = np.nanargmin(np.abs(fn - f))
    return 100 * (f[i] + fn[i]) / 2

for name, split in [("dev_scores.csv", "dev"), ("final_test_scores.csv", "track-1_test")]:
    p = os.path.join(run, name)
    if not os.path.exists(p):
        print(split, "no scores yet"); continue
    s = pd.read_csv(p)
    m = pd.read_csv(os.path.join(M, f"{split}.csv"))
    m = m[m["keep"].astype(bool)]
    d = m.merge(s, on="id")
    print(f"{split}: {len(s):,} scores | {len(d):,} matched of {len(m):,} | dup ids {s.id.duplicated().sum()} | NaN {s.score.isna().sum()} | EER {eer(d):.3f}%")
    print(f"   mean score: bona fide {d[d.label == 1].score.mean():.2f} | spoof {d[d.label == 0].score.mean():.2f}")
