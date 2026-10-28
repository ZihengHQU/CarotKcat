# -*- coding: utf-8 -*-
"""Step 17: add bootstrap CIs and paired tests for the CarotKcat-GNN variant,
then merge it into clean_eval.csv / benchmark_table.csv."""
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, r2_score

def metrics(y, p):
    return {"Pearson": float(pearsonr(y, p)[0]), "R2": float(r2_score(y, p)),
            "RMSE": float(np.sqrt(mean_squared_error(y, p)))}

def boot(y, p, n=1000, seed=3):
    rng = np.random.default_rng(seed); N = len(y)
    out = {"Pearson": [], "R2": [], "RMSE": []}
    for _ in range(n):
        i = rng.integers(0, N, N)
        if len(np.unique(i)) < 10 or np.std(p[i]) < 1e-12:
            continue
        out["Pearson"].append(pearsonr(y[i], p[i])[0])
        out["R2"].append(r2_score(y[i], p[i]))
        out["RMSE"].append(np.sqrt(mean_squared_error(y[i], p[i])))
    return {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in out.items()}

def paired(y, a, b, n=2000, seed=4):
    rng = np.random.default_rng(seed); N = len(y); d = []
    for _ in range(n):
        i = rng.integers(0, N, N)
        if len(np.unique(i)) < 10:
            continue
        d.append(pearsonr(y[i], a[i])[0] - pearsonr(y[i], b[i])[0])
    d = np.asarray(d)
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return float(d.mean()), float(min(max(p, 0.0), 1.0))

rows, ptest = [], []
for proto in ["random", "group"]:
    z = np.load(f"results/clean_preds_{proto}.npz", allow_pickle=True)
    y = z["y"]
    g = np.load(f"results/gnn_pred_{proto}.npy")
    assert len(g) == len(y)
    m = metrics(y, g); ci = boot(y, g)
    row = {"protocol": proto, "model": "CarotKcat-GNN (ESM+GNN substrate)", **m}
    for k in ("Pearson", "R2", "RMSE"):
        row[f"{k}_lo"], row[f"{k}_hi"] = ci[k]
    rows.append(row)
    print(proto, {k: round(v, 3) for k, v in m.items()})
    for ref_name in ["CarotKcat-ensemble", "CarotKcat (dual-tower)"]:
        d, p = paired(y, g, z[ref_name])
        ptest.append({"protocol": proto,
                      "comparison": f"CarotKcat-GNN vs {ref_name}",
                      "delta_Pearson": d, "p_value": p})
        print(f"  vs {ref_name}: dP={d:+.3f} p={p:.3f}")

gnn = pd.DataFrame(rows)
ce = pd.read_csv("results/clean_eval.csv")
ce = ce[ce["model"] != "CarotKcat-GNN (ESM+GNN substrate)"]
ce = pd.concat([ce, gnn], ignore_index=True)
ce.to_csv("results/clean_eval.csv", index=False)

bt = pd.read_csv("results/benchmark_table.csv")
bt = bt[bt["model"] != "CarotKcat-GNN (ESM+GNN substrate)"]
bt = pd.concat([bt, gnn[["protocol", "model", "Pearson", "R2", "RMSE",
                         "Pearson_lo", "Pearson_hi", "R2_lo", "R2_hi",
                         "RMSE_lo", "RMSE_hi"]]], ignore_index=True)
bt.to_csv("results/benchmark_table.csv", index=False)

pt = pd.read_csv("results/paired_tests.csv")
pt = pd.concat([pt, pd.DataFrame(ptest)], ignore_index=True)
pt.to_csv("results/paired_tests.csv", index=False)
print("merged CarotKcat-GNN into clean_eval.csv / benchmark_table.csv / paired_tests.csv")
