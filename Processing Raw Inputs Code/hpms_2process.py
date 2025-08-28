#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HPMS 5s Preprocessor (Group-aware)
----------------------------------
- Resamples raw rows into 5-second bins
- Interprets multi-column parameter groups into ONE parent column:
    * room_noise    <- (rms, noise_level)
    * mouse_movement<- (X, Y, Speed)
    * body_posture  <- (left/right shoulder, nose xyz, shoulder_diff, head_forward, posture, bad_posture_streak)
    * body_movement <- (depth_change, movement_status)
    * voice         <- (Pitch (Hz), Jitter, Shimmer, Voice State, WPM, Avg Sentence Length, Stutters, Clarity)
- Interprets eye_tracking from raw gaze tokens (CENTER/LEFT/RIGHT/BLINKING…)
- Keeps environment + vitals numeric means
- Keeps face_emotion, face_stress, task_timestamp as modes

Usage:
    python hpms_5s_preprocess_v2.py "Aug 13 Scenario Data - Sheet1 (1).csv" "Aug13_HPMS_5s_v2.csv"
"""

import pandas as pd
import numpy as np
from typing import Optional
from pathlib import Path
import re
from collections import Counter
# ------------------------ Column discovery ------------------------

def norm_name(s: str) -> str:
    return s.strip().lower().replace("-", "_").replace(" ", "_")

def find_timestamp_col(df: pd.DataFrame) -> Optional[str]:
    candidates = [
        "timestamp","time","datetime","date_time",
        "current_task_timestamp","task_timestamp",
        "current task timestamp","task timestamp","Task Timestamp"
    ]
    for c in df.columns:
        if norm_name(c) in [norm_name(x) for x in candidates]:
            return c
    return None

def find_col(df: pd.DataFrame, keys) -> Optional[str]:
    if isinstance(keys, str):
        keys = [keys]
    targets = [norm_name(k) for k in keys]
    for c in df.columns:
        if norm_name(c) in targets:
            return c
    for c in df.columns:
        nc = norm_name(c)
        if any(t in nc for t in targets):
            return c
    return None

# ------------------------ Interpreters ------------------------

def mode_or_unknown(s: pd.Series, unknown="unknown"):
    s = s.dropna()
    if s.empty:
        return unknown
    m = s.mode()
    return str(m.iloc[0]) if not m.empty else unknown





# def interpret_eye_tracking(tokens: pd.Series) -> str:
#     if tokens is None or tokens.empty:
#         return "unknown"
#     toks = [str(x).lower() for x in tokens.dropna()]
#     if not toks:
#         return "unknown"
#     total = max(len(toks), 1)
#     blink_frac = sum(("blink" in t) for t in toks) / total
#     center_frac = sum(("center" in t) for t in toks) / total
#     left_right_frac = sum(("left" in t or "right" in t) for t in toks) / total
#     if blink_frac >= 0.30:
#         return "blinking_frequently"
#     if center_frac >= 0.60:
#         return "focused"
#     if left_right_frac >= 0.40:
#         return "distracted"
#     return "distracted"




def _normalize_gaze_token(tok: str) -> str:
    if tok is None:
        return "unknown"
    t = str(tok).strip().lower()
    # accept a few common literal forms
    if t in {"center", "centre"}: return "center"
    if t in {"left"}: return "left"
    if t in {"right"}: return "right"
    if t in {"blink", "blinking", "blinked"}: return "blink"
    # handle combined tokens like "center_left", "left-right", etc.
    parts = re.split(r"[^\w]+", t)
    parts = [p for p in parts if p]
    parts_set = set(parts)
    if "blink" in parts_set: return "blink"
    if "center" in parts_set or "centre" in parts_set:
        # if combined with left/right, you can choose policy; here we favor center
        return "center"
    if "left" in parts_set: return "left"
    if "right" in parts_set: return "right"
    return "unknown"

def interpret_eye_tracking(tokens: pd.Series,
                           center_thr=0.50,
                           blink_thr=0.20,
                           lr_thr=0.50) -> str:
    """Robust 5s interpretation: focused / blinking_frequently / distracted."""
    if tokens is None or tokens.dropna().empty:
        return "unknown"
    norm = tokens.dropna().map(_normalize_gaze_token)
    total = len(norm)
    counts = Counter(norm)
    center_frac = counts.get("center", 0) / total
    blink_frac  = counts.get("blink", 0) / total
    lr_frac     = (counts.get("left", 0) + counts.get("right", 0)) / total

    # priority: blinks > focused > distracted
    if blink_frac >= blink_thr:
        return "blinking_frequently"
    if center_frac >= center_thr:
        return "focused"
    if lr_frac >= lr_thr:
        return "distracted"
    # fallback: use plurality
    label = counts.most_common(1)[0][0]
    return {"center":"focused", "blink":"blinking_frequently"}.get(label, "distracted")








def interpret_room_noise(df_chunk: pd.DataFrame) -> str:
    c_noise = find_col(df_chunk, "noise_level")
    if c_noise:
        v = pd.to_numeric(df_chunk[c_noise], errors="coerce").mean()
        if pd.isna(v): return "unknown"
        if v < 40: return "quiet"
        if v < 65: return "normal"
        return "loud"
    c_rms = find_col(df_chunk, "rms")
    if c_rms:
        v = pd.to_numeric(df_chunk[c_rms], errors="coerce").mean()
        if pd.isna(v): return "unknown"
        if v < 0.01: return "quiet"
        if v < 0.05: return "normal"
        return "loud"
    return "unknown"

def interpret_mouse(df_chunk: pd.DataFrame) -> str:
    c_speed = find_col(df_chunk, "Speed")
    if c_speed:
        v = pd.to_numeric(df_chunk[c_speed], errors="coerce").mean()
        if pd.isna(v): return "unknown"
        return "active" if v >= 5.0 else "calm"
    return "unknown"

def interpret_body_posture(df_chunk: pd.DataFrame) -> str:
    c_posture = find_col(df_chunk, "posture")
    if c_posture:
        label = mode_or_unknown(df_chunk[c_posture])
        if "good" in label.lower(): return "good"
        if any(k in label.lower() for k in ["bad","poor","slouch"]): return "poor"
    c_head = find_col(df_chunk, "head_forward")
    if c_head:
        mean_val = pd.to_numeric(df_chunk[c_head], errors="coerce").fillna(0).mean()
        if mean_val >= 0.5: return "head_forward"
    return "unknown"

def interpret_body_movement(df_chunk: pd.DataFrame) -> str:
    c_status = find_col(df_chunk, "movement_status")
    if c_status:
        vals = [str(x).lower() for x in df_chunk[c_status].dropna()]
        if vals:
            frac_move = sum(("move" in v or "active" in v or "walking" in v) for v in vals)/len(vals)
            return "moving" if frac_move > 0.30 else "still"
    c_depth = find_col(df_chunk, "depth_change")
    if c_depth:
        v = pd.to_numeric(df_chunk[c_depth], errors="coerce").abs().mean()
        if pd.notna(v) and v > 0.5: return "moving"
    return "still"

def interpret_voice(df_chunk: pd.DataFrame) -> str:
    c_state = find_col(df_chunk, "voice state")
    if c_state:
        vals = [str(x).lower() for x in df_chunk[c_state].dropna()]
        if any("talk" in v or "speak" in v for v in vals): return "talking"
    c_wpm = find_col(df_chunk, "WPM")
    if c_wpm:
        v = pd.to_numeric(df_chunk[c_wpm], errors="coerce").mean()
        if pd.notna(v) and v >= 10: return "talking"
    return "silent"

# ------------------------ Pipeline ------------------------

def preprocess_to_5s_groupaware(input_csv: str, output_csv: str) -> pd.DataFrame:
    df = pd.read_csv(input_csv)

    ts_col = find_timestamp_col(df)
    if ts_col is None:
        raise ValueError("No timestamp column found")
    df["timestamp"] = pd.to_datetime(df[ts_col], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    df["t_5s"] = df["timestamp"].dt.floor("5S")

    # Environment/vitals means
    keep_cols = {}
    for k,new in [
        ("temperature_C","room_temperature"),
        ("pressure_hPa","room_pressure"),
        ("humidity_%","room_humidity"),
        ("light_lux","room_light_intensity"),
        ("skin_temperature_C","skin_temperature"),
        ("CCT_K","light_temperature"),
        ("Heart Rate (bpm)","heart_rate"),
    ]:
        c = find_col(df, k)
        if c: keep_cols[c] = new
    grouped = df[["t_5s"]+list(keep_cols.keys())].groupby("t_5s").mean().reset_index()
    grouped.rename(columns=keep_cols, inplace=True)

    # Add interpreted groups
    grouped = grouped.merge(df.groupby("t_5s").apply(interpret_eye_tracking).reset_index(name="eye_tracking"), on="t_5s")
    grouped = grouped.merge(df.groupby("t_5s").apply(interpret_room_noise).reset_index(name="room_noise"), on="t_5s")
    grouped = grouped.merge(df.groupby("t_5s").apply(interpret_mouse).reset_index(name="mouse_movement"), on="t_5s")
    grouped = grouped.merge(df.groupby("t_5s").apply(interpret_body_posture).reset_index(name="body_posture"), on="t_5s")
    grouped = grouped.merge(df.groupby("t_5s").apply(interpret_body_movement).reset_index(name="body_movement"), on="t_5s")
    grouped = grouped.merge(df.groupby("t_5s").apply(interpret_voice).reset_index(name="voice"), on="t_5s")

    # Face emotion, stress, task timestamp
    col_emotion = find_col(df, "Emotion Detection")
    if col_emotion:
        emo = df.groupby("t_5s")[col_emotion].apply(mode_or_unknown).reset_index(name="face_emotion")
        grouped = grouped.merge(emo, on="t_5s")
    col_stress = find_col(df, "face_stress")
    if col_stress:
        st = df.groupby("t_5s")[col_stress].apply(mode_or_unknown).reset_index(name="face_stress")
        grouped = grouped.merge(st, on="t_5s")
    col_taskts = find_col(df, "task timestamp")
    if col_taskts:
        tt = df.groupby("t_5s")[col_taskts].apply(mode_or_unknown).reset_index(name="task_timestamp")
        grouped = grouped.merge(tt, on="t_5s")

    grouped.rename(columns={"t_5s":"timestamp"}, inplace=True)
    grouped.to_csv(output_csv, index=False)
    return grouped

# ------------------------ CLI ------------------------

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Condense & interpret raw HPMS CSV into 5-second parent-parameter columns.")
    ap.add_argument("input_csv", help="Path to raw input CSV")
    ap.add_argument("output_csv", help="Path to output CSV (5-second interpreted)")
    args = ap.parse_args()
    out = preprocess_to_5s_groupaware(args.input_csv, args.output_csv)
    print(f"Wrote {args.output_csv} with {len(out)} rows and {len(out.columns)} columns.")
