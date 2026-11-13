"""
Short-horizon dissolved oxygen (DO) forecasting for Patapsco River - Aquarium East Bottom.

Put this file in the EDA folder next to Test_Station.csv, then run it from a
terminal opened in EDA:
    python aeb_do_pipeline.py

Needs: pandas, numpy, scikit-learn, matplotlib
Output: EDA/results/ (JSON/CSV tables and a forecast plot)
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# ---------------- settings ----------------
CSV_PATH = "Test_Station.csv"
OUT_DIR = "results"
HORIZONS = {"1h": 4, "3h": 12, "6h": 24, "12h": 48, "24h": 96}  # 15-min rows per horizon
HYPOXIA = 2.0          # mg/L threshold
TRAIN_FRAC = 0.70      # first 70% of rows train, last 30% test
RANDOM_STATE = 0


# ---------------- helpers ----------------
def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mae(a, b):
    return float(np.mean(np.abs(a - b)))


# ---------------- load and check ----------------
os.makedirs(OUT_DIR, exist_ok=True)
raw = pd.read_csv(CSV_PATH)

# Column names as they appear in the raw 'Continuous Measurement' download
df = pd.DataFrame({
    "date": pd.to_datetime(raw["DateTime"], format="%m/%d/%Y %H:%M", errors="coerce"),
    "do": pd.to_numeric(raw["DO_mg/L"], errors="coerce"),
    "do_sat": pd.to_numeric(raw["DO_pctSat"], errors="coerce"),
    "sal": pd.to_numeric(raw["Salinity_ppt"], errors="coerce"),
    "ph": pd.to_numeric(raw["pH"], errors="coerce"),
    "turb": pd.to_numeric(raw["Turb_NTU"], errors="coerce"),
    "chl": pd.to_numeric(raw["Chl_ug/L"], errors="coerce"),
    "temp": pd.to_numeric(raw["Temp_C"], errors="coerce"),
})
n = len(df)

checks = {"rows": n, "rows_with_do": int(df["do"].notna().sum()),
          "expected_rows_apr1_oct1_2024": 17664}
valid_dates = df["date"].notna().mean()
checks["valid_date_share"] = round(float(valid_dates), 4)
if valid_dates >= 0.99:
    steps = df["date"].diff().dropna()
    checks["gaps_not_15min"] = int((steps != pd.Timedelta(minutes=15)).sum())
    checks["date_min"] = str(df["date"].min())
    checks["date_max"] = str(df["date"].max())
    if checks["gaps_not_15min"] > 0:
        print("WARNING: some rows are not exactly 15 minutes apart. Horizons are row shifts, "
              "so they will be approximate around those gaps.")
else:
    print("WARNING: Date column is not usable. Row order will be treated as time.")

if n != checks["expected_rows_apr1_oct1_2024"]:
    print(f"NOTE: {n} rows, README expects 17664. Check the download date range.")

# EDA checks that the README asks for
checks["do_negative"] = int((df["do"] < 0).sum())
checks["do_hypoxic_lt2"] = int((df["do"] < HYPOXIA).sum())
checks["missing_pct"] = {c: round(100 * df[c].isna().mean(), 2)
                         for c in ["sal", "ph", "do", "do_sat", "turb", "chl", "temp"]}
checks["corr_do_sat_vs_do"] = round(float(df["do"].corr(df["do_sat"])), 3)
checks["corr_sat_vs_ph"] = round(float(df["do_sat"].corr(df["ph"])), 3)
checks["acf_6h"] = round(float(df["do"].autocorr(24)), 3)
checks["acf_12h"] = round(float(df["do"].autocorr(48)), 3)
checks["acf_24h"] = round(float(df["do"].autocorr(96)), 3)

# README: keep rows with missing values; clip small negatives to zero
df["do"] = df["do"].clip(lower=0)
df["do_sat"] = df["do_sat"].clip(lower=0)

# ---------------- features (past and present only) ----------------
X = pd.DataFrame(index=df.index)
for c in ["do", "sal", "ph", "turb", "chl", "temp"]:
    X[c] = df[c]
for lag in [1, 2, 4, 8, 16, 32]:
    X[f"do_lag{lag}"] = df["do"].shift(lag)
for c in ["sal", "ph", "turb", "chl", "temp"]:
    for lag in [4, 16]:
        X[f"{c}_lag{lag}"] = df[c].shift(lag)
X["do_diff1"] = df["do"] - df["do"].shift(1)
X["do_diff4"] = df["do"] - df["do"].shift(4)
X["temp_diff4"] = df["temp"] - df["temp"].shift(4)
X["do_roll16_mean"] = df["do"].rolling(16, min_periods=8).mean()
X["do_roll96_mean"] = df["do"].rolling(96, min_periods=48).mean()
X = X.astype(float).values

current = df["do"].values
idx = np.arange(n)
split = int(TRAIN_FRAC * n)


def make_models():
    return {
        "ridge": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1.0)),
        "hist_gbm": HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05,
                                                  random_state=RANDOM_STATE),
    }


# ---------------- supervised: time-ordered split ----------------
results = []
hypoxia_rows = []
plot_store = None

for hname, h in HORIZONS.items():
    y = df["do"].shift(-h).values
    valid = ~np.isnan(y) & ~np.isnan(current)
    tr = (idx < split - h) & valid      # drop training targets that reach into the test period
    te = (idx >= split) & valid
    y_te = y[te]

    preds = {"persistence": current[te]}
    for name, model in make_models().items():
        model.fit(X[tr], (y - current)[tr])          # learn the change, add it back
        preds[name] = current[te] + model.predict(X[te])

    base = rmse(y_te, preds["persistence"])
    row = {"horizon": hname, "n_train": int(tr.sum()), "n_test": int(te.sum())}
    for name, p in preds.items():
        r = rmse(y_te, p)
        row[name] = {
            "rmse": round(r, 4),
            "mae": round(mae(y_te, p), 4),
            "improvement_vs_persistence_pct": round(100 * (base - r) / base, 1),
            "meets_20pct_target": bool(r <= 0.8 * base),
        }
    results.append(row)

    actual_hyp = (y_te < HYPOXIA).astype(int)
    for name, p in preds.items():
        pred_hyp = (p < HYPOXIA).astype(int)
        hypoxia_rows.append({
            "horizon": hname, "model": name,
            "precision": round(precision_score(actual_hyp, pred_hyp, zero_division=0), 3),
            "recall": round(recall_score(actual_hyp, pred_hyp, zero_division=0), 3),
            "f1": round(f1_score(actual_hyp, pred_hyp, zero_division=0), 3),
        })

    if hname == "6h":
        plot_store = (y_te, preds)

# ---------------- forward-chaining CV (4 folds) ----------------
cv_rows = []
bounds = np.linspace(0.5 * n, n, 5).astype(int)
for hname, h in HORIZONS.items():
    y = df["do"].shift(-h).values
    valid = ~np.isnan(y) & ~np.isnan(current)
    folds = []
    for k in range(4):
        start, stop = bounds[k], bounds[k + 1]
        tr = (idx < start - h) & valid
        te = (idx >= start) & (idx < stop) & valid
        m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=RANDOM_STATE)
        m.fit(X[tr], (y - current)[tr])
        p = current[te] + m.predict(X[te])
        folds.append({"fold": k + 1,
                      "gbm_rmse": round(rmse(y[te], p), 3),
                      "persistence_rmse": round(rmse(y[te], current[te]), 3)})
    cv_rows.append({"horizon": hname, "folds": folds,
                    "mean_gbm": round(float(np.mean([f["gbm_rmse"] for f in folds])), 3),
                    "mean_persistence": round(float(np.mean([f["persistence_rmse"] for f in folds])), 3)})

# ---------------- unsupervised: PCA and clustering ----------------
state_cols = ["do", "sal", "ph", "turb", "chl", "temp"]
S = df[state_cols].dropna()
Z = StandardScaler().fit_transform(S)
pca = PCA(n_components=4, random_state=RANDOM_STATE).fit(Z)
km = KMeans(n_clusters=4, n_init=10, random_state=RANDOM_STATE).fit(Z)
S = S.assign(cluster=km.labels_)

clusters = []
for c in range(4):
    sub = S[S.cluster == c]
    clusters.append({
        "cluster": c,
        "share_pct": round(100 * len(sub) / len(S), 1),
        "mean_do": round(float(sub["do"].mean()), 2),
        "mean_temp": round(float(sub["temp"].mean()), 2),
        "mean_sal": round(float(sub["sal"].mean()), 2),
        "pct_hypoxic": round(100 * float((sub["do"] < HYPOXIA).mean()), 1),
    })
unsup = {
    "pca_explained_variance_ratio": [round(float(v), 3) for v in pca.explained_variance_ratio_],
    "pca_loadings_pc1": dict(zip(state_cols, [round(float(v), 3) for v in pca.components_[0]])),
    "clusters": clusters,
}

# ---------------- save ----------------
with open(f"{OUT_DIR}/supervised_results.json", "w") as f:
    json.dump(results, f, indent=2)
pd.DataFrame(hypoxia_rows).to_csv(f"{OUT_DIR}/hypoxia_detection.csv", index=False)
with open(f"{OUT_DIR}/cv_results.json", "w") as f:
    json.dump(cv_rows, f, indent=2)
with open(f"{OUT_DIR}/unsupervised_results.json", "w") as f:
    json.dump(unsup, f, indent=2)
with open(f"{OUT_DIR}/checks.json", "w") as f:
    json.dump(checks, f, indent=2)

# 6-hour forecast plot, first 14 days of the test period
y_te, preds = plot_store
span = slice(0, min(14 * 96, len(y_te)))
fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(y_te[span], color="black", lw=1, label="actual DO 6 h ahead")
ax.plot(preds["persistence"][span], color="grey", lw=1, ls="--", label="persistence")
ax.plot(preds["hist_gbm"][span], color="tab:blue", lw=1, label="gradient boosting")
ax.axhline(HYPOXIA, color="red", lw=0.8, ls=":", label="hypoxia 2 mg/L")
ax.set_ylabel("DO (mg/L)")
ax.set_xlabel("15-min steps into test period")
ax.set_title("6-hour-ahead DO forecast, first 14 days of test period")
ax.legend(loc="upper right", fontsize=8)
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/forecast_6h.png", dpi=120)

# ---------------- print summary ----------------
print("\nDATA CHECKS")
print(json.dumps(checks, indent=2))
print("\nFORECAST RESULTS (RMSE mg/L on last 30%; improvement vs persistence)")
print(f"{'horizon':>8} {'persist':>9} {'ridge':>9} {'ridge %':>9} {'gbm':>9} {'gbm %':>9}")
for r in results:
    print(f"{r['horizon']:>8} {r['persistence']['rmse']:>9.3f} {r['ridge']['rmse']:>9.3f} "
          f"{r['ridge']['improvement_vs_persistence_pct']:>9.1f} {r['hist_gbm']['rmse']:>9.3f} "
          f"{r['hist_gbm']['improvement_vs_persistence_pct']:>9.1f}")
print(f"\nSaved results to {OUT_DIR}/")