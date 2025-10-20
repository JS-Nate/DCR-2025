# dcr_viz.py
# Ready-to-run visuals for your Aug 13 scenario.
# Requires: pandas, numpy, matplotlib (no other libs).

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta

CSV_PATH = "Aug13_HPMS_5s_v2.csv"

# ---------------------------
# 1) LOAD & PREP
# ---------------------------
df = pd.read_csv(CSV_PATH)
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
df = df.sort_values("timestamp").reset_index(drop=True)

# Ensure expected columns exist (fallbacks if missing)
for col in [
    "heart_rate","skin_temperature",
    "room_temperature","room_humidity","room_light_intensity",
    "eye_tracking","body_posture","task_timestamp","task"
]:
    if col not in df.columns:
        df[col] = np.nan

df["task_timestamp"] = df["task_timestamp"].fillna("unknown")
df["task"] = df["task"].fillna("")

# Identify SMR process “event” rows where a Step begins (based on task_timestamp)
event_mask = df["task_timestamp"].str.contains("Step", case=False, na=False)
events = df.loc[event_mask, ["timestamp","task_timestamp"]].copy()
# Short labels like "S1", "S2" ...
events["label"] = [f"S{i+1}" for i in range(len(events))]

# ---------------------------
# 2) DERIVED METRICS (Human)
# ---------------------------
# Rolling baseline HR (60 s window, your data is at ~5s cadence => ~12 samples)
# Use median for stability. If too short, adapt to min periods.
win = 12
hr = df["heart_rate"].astype(float)
df["hr_baseline"] = hr.rolling(window=win, min_periods=max(3,win//3)).median()
df["hr_baseline"].fillna(method="bfill", inplace=True)
df["workload_index"] = (hr - df["hr_baseline"]) / df["hr_baseline"]
df["workload_index"] = df["workload_index"].clip(lower=0.0)  # only show positive load

# Attention proxy from eye_tracking text (robust to your categorical strings)
def attention_from_eye(s: str) -> float:
    if not isinstance(s, str):
        return 0.6
    st = s.lower()
    # Tunable mapping; conservative defaults
    if "center" in st or "focused" in st or "focus" in st:
        return 0.9
    if "distract" in st:
        return 0.35
    if "blink" in st:
        return 0.55
    if "unknown" in st:
        return 0.6
    return 0.6

df["attention_score"] = df["eye_tracking"].apply(attention_from_eye)

# Posture quality (good/fair/poor → 1.0/0.6/0.35)
def posture_to_score(s: str) -> float:
    if not isinstance(s, str):
        return 0.6
    st = s.lower().strip()
    if "good" in st:
        return 0.9
    if "fair" in st:
        return 0.6
    if "poor" in st:
        return 0.35
    return 0.6

df["posture_score"] = df["body_posture"].apply(posture_to_score)

# Composite “HPMS load” (simple, transparent weighting you can change)
# Higher = more support needed
# Workload ↑ when HR rises; Attention ↓ when distracted; Posture ↓ when poor
df["hpms_load"] = (
    0.6 * df["workload_index"].fillna(0) +
    0.25 * (1.0 - df["attention_score"]) +
    0.15 * (1.0 - df["posture_score"])
)

# ---------------------------
# 3) NORMALIZATION HELPERS
# ---------------------------
def norm(s):
    s = pd.to_numeric(s, errors="coerce")
    return (s - s.min()) / (s.max() - s.min()) if s.max() != s.min() else s*0

df["hr_norm"]     = norm(df["heart_rate"])
df["skin_norm"]   = norm(df["skin_temperature"])
df["roomT_norm"]  = norm(df["room_temperature"])
df["humid_norm"]  = norm(df["room_humidity"])
df["light_norm"]  = norm(df["room_light_intensity"])
df["hpms_norm"]   = norm(df["hpms_load"])

# ---------------------------
# 4) FIGURE: 3-TIER INTEGRATED CHART
# ---------------------------
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

# Tier 1: Human
axes[0].plot(df["timestamp"], df["hr_norm"], label="Heart rate (norm)")
axes[0].plot(df["timestamp"], df["skin_norm"], label="Skin temp (norm)")
axes[0].plot(df["timestamp"], df["hpms_norm"], label="HPMS load (norm)", linestyle="--")
axes[0].set_title("Tier 1 — Human (Workload & Arousal)")
axes[0].set_ylabel("Normalized")
axes[0].grid(True)
axes[0].legend(loc="upper left")

# Tier 2: Control Room Environment
axes[1].plot(df["timestamp"], df["roomT_norm"], label="Room temp (norm)")
axes[1].plot(df["timestamp"], df["humid_norm"], label="Humidity (norm)")
axes[1].plot(df["timestamp"], df["light_norm"], label="Light intensity (norm)")
axes[1].set_title("Tier 2 — Control Room Environment")
axes[1].set_ylabel("Normalized")
axes[1].grid(True)
axes[1].legend(loc="upper left")

# Tier 3: SMR Process (timeline via vertical lines + labels)
axes[2].plot(df["timestamp"], np.zeros(len(df)), alpha=0)  # empty base
for _, r in events.iterrows():
    axes[2].axvline(r["timestamp"], linestyle="--", alpha=0.6)
    axes[2].text(r["timestamp"], 0.5, r["label"], rotation=90,
                 va="center", ha="center", fontsize=8)
# Optional: show any non-empty task text as callouts (short)
for i, r in df[df["task"].str.len() > 0].iloc[::40].iterrows():
    axes[2].text(r["timestamp"], 0.15, "task", rotation=90, fontsize=7, alpha=0.7)
axes[2].set_title("Tier 3 — SMR Process Timeline (Simulator Tasks)")
axes[2].set_yticks([])
axes[2].set_xlabel("Time")
axes[2].grid(True, axis="x")

# Sync event markers across tiers
for ax in axes[:2]:
    for _, r in events.iterrows():
        ax.axvline(r["timestamp"], linestyle="--", alpha=0.25)

fig.tight_layout()
fig.savefig("fig_3tier.png", dpi=200)

# ---------------------------
# 5) FIGURE: PERFORMANCE HEATMAP (metrics × time bins)
# ---------------------------
# Discretize into N bins along time for compact view
N = 24
df["bin"] = pd.qcut(df.index, q=N, labels=False, duplicates="drop")

heat = df.groupby("bin").agg({
    "workload_index":"mean",
    "attention_score":"mean",
    "posture_score":"mean",
    "room_temperature":"mean",
    "room_humidity":"mean",
    "room_light_intensity":"mean"
}).reset_index()

# Normalize each row for heat display across different scales
heat_norm = heat.copy()
for c in ["workload_index","attention_score","posture_score",
          "room_temperature","room_humidity","room_light_intensity"]:
    heat_norm[c] = norm(heat[c])

heat_matrix = heat_norm[[
    "workload_index","attention_score","posture_score",
    "room_temperature","room_humidity","room_light_intensity"
]].T.values

fig2 = plt.figure(figsize=(12, 4))
ax2 = fig2.add_subplot(111)
im = ax2.imshow(heat_matrix, aspect="auto", interpolation="nearest")
ax2.set_yticks(range(6))
ax2.set_yticklabels([
    "Workload ↑",
    "Attention ↑",
    "Posture ↑",
    "Room Temp",
    "Humidity",
    "Light"
])
ax2.set_xticks(range(0, heat_norm.shape[0], max(1, heat_norm.shape[0]//8)))
ax2.set_xticklabels([str(i) for i in range(0, heat_norm.shape[0], max(1, heat_norm.shape[0]//8))])
ax2.set_title("Human–Control Room Heatmap (normalized 0–1 by row)")
fig2.colorbar(im, ax=ax2, fraction=0.025, pad=0.02)
fig2.tight_layout()
fig2.savefig("fig_heatmap.png", dpi=200)

# ---------------------------
# 6) FIGURE: DCR ACTION TRIGGERS
# ---------------------------
# Simple rule: trigger adaptive response if workload above 0.12 AND attention below 0.6
trigger = (df["workload_index"] > 0.12) & (df["attention_score"] < 0.6)
df["trigger"] = trigger

fig3, ax3 = plt.subplots(figsize=(12, 4))
ax3.plot(df["timestamp"], df["workload_index"], label="Workload index")
ax3.plot(df["timestamp"], df["attention_score"], label="Attention score")
ax3.fill_between(df["timestamp"], 0, 1,
                 where=df["trigger"],
                 alpha=0.2, transform=ax3.get_xaxis_transform(),
                 label="DCR adaptive trigger")
# Threshold guides
ax3.axhline(0.12, linestyle="--", alpha=0.5)
ax3.axhline(0.6, linestyle="--", alpha=0.5)
ax3.set_title("When would the DCR adapt? (Trigger bands)")
ax3.set_ylabel("Score")
ax3.set_xlabel("Time")
ax3.legend(loc="upper left")
ax3.grid(True)
fig3.tight_layout()
fig3.savefig("fig_triggers.png", dpi=200)

print("Saved: fig_3tier.png, fig_heatmap.png, fig_triggers.png")
# ---------------------------
# END