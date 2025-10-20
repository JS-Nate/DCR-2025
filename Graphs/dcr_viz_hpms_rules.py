# dcr_viz_hpms_rules.py
# Ready-to-run visuals + HPMS-rule triggers (configurable).
# Requires: pandas, numpy, matplotlib.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta

CSV_PATH = "Aug13_HPMS_5s_v2.csv"

# ========= HPMS CONFIG (edit these to match your official thresholds) =========
CFG = dict(
    # Rolling baselines & windows
    cadence_s=5,                         # your data cadence (~5 s)
    hr_baseline_window_s=60,             # rolling median window
    # Workload (WI) zones relative to rolling baseline
    wi_mild=0.05,                        # 5–12% above baseline = mild
    wi_moderate=0.12,                    # >=12% = moderate
    wi_high=0.20,                        # >=20% = high
    # Attention & posture scoring cutoffs (0..1 scales after mapping)
    attention_low=0.55,                  # <0.55 => low attention
    posture_low=0.50,                    # <0.50 => poor posture
    # Environmental targets (used only for labels/interpretation)
    roomT_ok=(22.0, 24.5),               # °C
    humidity_ok=(40.0, 60.0),            # %
    light_ok=(100.0, 200.0),             # lux
    # Trigger persistence (avoid flicker)
    min_consec_points=3,                 # require N consecutive samples
    # Actions (enable/disable)
    enable_actions=True
)

# Actions (the logic is simple & transparent; adjust if needed)
# A1: Adaptive UI simplify
# A2: Lighting reduction (~10%)
# A3: Focus assist/highlight critical panel
# A4: Microbreak prompt (20–30 s)
# These are fired based on WI + attention + posture combos.
# ==============================================================================

def main():
    df = pd.read_csv(CSV_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Keep handy refs
    need_cols = [
        "heart_rate","skin_temperature","room_temperature",
        "room_humidity","room_light_intensity","eye_tracking",
        "body_posture","task_timestamp","task"
    ]
    for c in need_cols:
        if c not in df.columns: df[c] = np.nan
    df["task_timestamp"] = df["task_timestamp"].fillna("unknown")
    df["task"] = df["task"].fillna("")

    # --- SMR process events (Step markers)
    evt_mask = df["task_timestamp"].str.contains("Step", case=False, na=False)
    events = df.loc[evt_mask, ["timestamp","task_timestamp"]].copy()
    events["label"] = [f"S{i+1}" for i in range(len(events))]

    # --- Rolling baseline for HR → Workload Index (WI)
    win = max(3, int(round(CFG["hr_baseline_window_s"] / CFG["cadence_s"])))
    hr = pd.to_numeric(df["heart_rate"], errors="coerce")
    df["hr_baseline"] = hr.rolling(window=win, min_periods=max(3,win//3)).median()
    df["hr_baseline"] = df["hr_baseline"].bfill()
    df["workload_index"] = ((hr - df["hr_baseline"]) / df["hr_baseline"]).clip(lower=0.0).fillna(0.0)

    # --- Attention & posture scores from text fields
    def attention_from_eye(s):
        if not isinstance(s, str): return 0.6
        st = s.lower()
        if "center" in st or "focus" in st or "focused" in st: return 0.9
        if "distract" in st: return 0.35
        if "blink" in st: return 0.55
        if "unknown" in st: return 0.6
        return 0.6

    def posture_score(s):
        if not isinstance(s, str): return 0.6
        st = s.lower()
        if "good" in st: return 0.9
        if "fair" in st: return 0.6
        if "poor" in st: return 0.35
        return 0.6

    df["attention_score"] = df["eye_tracking"].apply(attention_from_eye)
    df["posture_score"]   = df["body_posture"].apply(posture_score)

    # --- Composite HPMS load (weights are transparent; tweak if needed)
    df["hpms_load"] = (
        0.6 * df["workload_index"] +
        0.25 * (1.0 - df["attention_score"]) +
        0.15 * (1.0 - df["posture_score"])
    )

    # --- Normalized series for plotting
    def norm(s):
        s = pd.to_numeric(s, errors="coerce")
        mx, mn = s.max(), s.min()
        return (s - mn) / (mx - mn) if mx != mn else s*0

    df["hr_norm"]    = norm(df["heart_rate"])
    df["skin_norm"]  = norm(df["skin_temperature"])
    df["roomT_norm"] = norm(df["room_temperature"])
    df["humid_norm"] = norm(df["room_humidity"])
    df["light_norm"] = norm(df["room_light_intensity"])
    df["hpms_norm"]  = norm(df["hpms_load"])

    # ----------------------- HPMS ACTION RULES -----------------------
    # helper for persistence
    def persistent(mask, k):
        # require k consecutive True; returns boolean mask where run-length>=k
        arr = mask.to_numpy().astype(int)
        run = np.zeros_like(arr)
        cnt = 0
        for i, v in enumerate(arr):
            cnt = cnt + 1 if v == 1 else 0
            run[i] = 1 if cnt >= k else 0
        return pd.Series(run.astype(bool), index=mask.index)

    WI = df["workload_index"]
    ATT = df["attention_score"]
    POS = df["posture_score"]
    LIGHT = pd.to_numeric(df["room_light_intensity"], errors="coerce")
    # zones
    wi_mild, wi_mod, wi_high = CFG["wi_mild"], CFG["wi_moderate"], CFG["wi_high"]

    # A1: Adaptive UI simplify -> WI >= moderate OR (ATT low & WI >= mild)
    a1_raw = (WI >= wi_mod) | ((ATT < CFG["attention_low"]) & (WI >= wi_mild))
    A1 = persistent(a1_raw, CFG["min_consec_points"])

    # A2: Lighting reduction (~10%) -> ATT low & LIGHT >= upper band of ok range
    light_hi = CFG["light_ok"][1]
    a2_raw = (ATT < CFG["attention_low"]) & (LIGHT >= light_hi)
    A2 = persistent(a2_raw, CFG["min_consec_points"])

    # A3: Focus assist/highlight panel -> ATT low OR POS low when WI >= mild
    a3_raw = ((ATT < CFG["attention_low"]) | (POS < CFG["posture_low"])) & (WI >= wi_mild)
    A3 = persistent(a3_raw, CFG["min_consec_points"])

    # A4: Microbreak prompt -> WI high sustained
    a4_raw = (WI >= wi_high)
    A4 = persistent(a4_raw, CFG["min_consec_points"])

    df["A1_UI_simplify"] = A1
    df["A2_light_dim"]   = A2
    df["A3_focus_assist"]= A3
    df["A4_microbreak"]  = A4

    # --- Produce trigger intervals CSV
    def intervals_from_mask(ts, mask, label):
        out = []
        active = False
        start = None
        for i in range(len(mask)):
            if mask.iat[i] and not active:
                active, start = True, ts.iat[i]
            if active and (i == len(mask)-1 or not mask.iat[i+1]):
                end = ts.iat[i]
                out.append(dict(action=label, start=start, end=end))
                active, start = False, None
        return out

    ts = df["timestamp"]
    rows = []
    for lbl, m in [("UI_simplify",A1),("Light_dim",A2),("Focus_assist",A3),("Microbreak",A4)]:
        rows += intervals_from_mask(ts, m, lbl)
    trig = pd.DataFrame(rows)
    if not trig.empty:
        trig["duration_s"] = (trig["end"] - trig["start"]).dt.total_seconds().astype(int)
        trig = trig.sort_values(["start","action"])
    trig.to_csv("trigger_intervals.csv", index=False)

    # --------------------------- FIGURES ---------------------------
    # 3-tier
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    axes[0].plot(df["timestamp"], df["hr_norm"], label="Heart rate (norm)")
    axes[0].plot(df["timestamp"], df["skin_norm"], label="Skin temp (norm)")
    axes[0].plot(df["timestamp"], df["hpms_norm"], "--", label="HPMS load (norm)")
    axes[0].set_title("Tier 1 — Human (Workload, Attention & Posture → HPMS load)")
    axes[0].set_ylabel("Normalized"); axes[0].legend(loc="upper left"); axes[0].grid(True)

    axes[1].plot(df["timestamp"], df["roomT_norm"], label="Room temp (norm)")
    axes[1].plot(df["timestamp"], df["humid_norm"], label="Humidity (norm)")
    axes[1].plot(df["timestamp"], df["light_norm"], label="Light intensity (norm)")
    axes[1].set_title("Tier 2 — Control Room Environment (stability bands)")
    axes[1].set_ylabel("Normalized"); axes[1].legend(loc="upper left"); axes[1].grid(True)

    # Mark SMR steps across tiers
    for ax in axes[:2]:
        for _, r in events.iterrows():
            ax.axvline(r["timestamp"], linestyle="--", alpha=0.25)
    axes[2].plot(df["timestamp"], np.zeros(len(df)), alpha=0)
    for _, r in events.iterrows():
        axes[2].axvline(r["timestamp"], linestyle="--", alpha=0.6)
        axes[2].text(r["timestamp"], 0.5, r["label"], rotation=90, va="center", ha="center", fontsize=8)
    axes[2].set_title("Tier 3 — SMR Process Timeline (Step markers)")
    axes[2].set_yticks([]); axes[2].set_xlabel("Time"); axes[2].grid(True, axis="x")

    fig.tight_layout(); fig.savefig("fig_3tier.png", dpi=200)

    # Heatmap
    N = 24
    df["bin"] = pd.qcut(df.index, q=N, labels=False, duplicates="drop")
    heat = df.groupby("bin").agg({
        "workload_index":"mean","attention_score":"mean","posture_score":"mean",
        "room_temperature":"mean","room_humidity":"mean","room_light_intensity":"mean"
    }).reset_index()
    def norm_col(s):
        mx, mn = s.max(), s.min()
        return (s - mn) / (mx - mn) if mx != mn else s*0
    for c in ["workload_index","attention_score","posture_score","room_temperature","room_humidity","room_light_intensity"]:
        heat[c] = norm_col(heat[c])
    mat = heat[["workload_index","attention_score","posture_score","room_temperature","room_humidity","room_light_intensity"]].T.values

    fig2 = plt.figure(figsize=(12,4)); ax2 = fig2.add_subplot(111)
    im = ax2.imshow(mat, aspect="auto", interpolation="nearest")
    ax2.set_yticks(range(6)); ax2.set_yticklabels(["Workload ↑","Attention ↑","Posture ↑","Room Temp","Humidity","Light"])
    ax2.set_title("Human–Control Room Heatmap (normalized per metric)")
    fig2.colorbar(im, ax=ax2, fraction=0.025, pad=0.02)
    fig2.tight_layout(); fig2.savefig("fig_heatmap.png", dpi=200)

    # Trigger plot
    fig3, ax3 = plt.subplots(figsize=(12,4))
    ax3.plot(df["timestamp"], df["workload_index"], label="Workload index")
    ax3.plot(df["timestamp"], df["attention_score"], label="Attention score")
    # shaded actions
    for lbl, m, alpha in [("UI simplify",A1,0.18),("Light dim",A2,0.18),("Focus assist",A3,0.12),("Microbreak",A4,0.12)]:
        ax3.fill_between(df["timestamp"], 0, 1, where=m, alpha=alpha, transform=ax3.get_xaxis_transform(), label=lbl)
    # thresholds
    ax3.axhline(CFG["wi_moderate"], linestyle="--", alpha=0.5)
    ax3.axhline(CFG["attention_low"], linestyle="--", alpha=0.5)
    ax3.set_title("HPMS Rule Triggers Over Time (bands show action windows)")
    ax3.set_ylabel("Score"); ax3.set_xlabel("Time"); ax3.legend(loc="upper left"); ax3.grid(True)
    fig3.tight_layout(); fig3.savefig("fig_triggers.png", dpi=200)

    print("Saved: fig_3tier.png, fig_heatmap.png, fig_triggers.png, trigger_intervals.csv")

if __name__ == "__main__":
    main()
