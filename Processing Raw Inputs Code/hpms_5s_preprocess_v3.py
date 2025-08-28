#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HPMS 5s Preprocessor (v3: robust eye-tracking + debug + smoothing)
------------------------------------------------------------------
- Resamples raw rows into 5-second bins
- Interprets multi-column groups into ONE parent parameter:
    * room_noise    <- (rms, noise_level)
    * mouse_movement<- (X, Y, Speed or mouse_movement categorical)
    * body_posture  <- (posture + head_forward + bad_posture_streak + kinematics)
    * body_movement <- (movement_status + depth_change)
    * voice         <- (Voice State / WPM)
- Eye-tracking: robust token parsing, tuned thresholds, per-bin diagnostics, optional smoothing
- Keeps environment + vitals numeric means
- Keeps face_emotion, face_stress, task_timestamp as modes
"""

import pandas as pd
import numpy as np
import re
from collections import Counter
from typing import Optional

# ------------------------ Utils ------------------------

def norm_name(s: str) -> str:
    return s.strip().lower().replace("-", "_").replace(" ", "_")

def find_timestamp_col(df: pd.DataFrame) -> Optional[str]:
    cands = [
        "timestamp","time","datetime","date_time",
        "current_task_timestamp","task_timestamp",
        "current task timestamp","task timestamp","Task Timestamp"
    ]
    for c in df.columns:
        if norm_name(c) in {norm_name(x) for x in cands}:
            return c
    return None

def find_col(df: pd.DataFrame, keys) -> Optional[str]:
    if isinstance(keys, str):
        keys = [keys]
    targets = {norm_name(k) for k in keys}
    for c in df.columns:
        if norm_name(c) in targets:
            return c
    for c in df.columns:
        nc = norm_name(c)
        if any(t in nc for t in targets):
            return c
    return None

def mode_or_unknown(s: pd.Series, unknown="unknown"):
    s = s.dropna()
    if s.empty:
        return unknown
    m = s.mode()
    return str(m.iloc[0]) if not m.empty else unknown

# ------------------------ Eye Tracking ------------------------

def _normalize_gaze_token(tok: str) -> str:
    if tok is None:
        return "unknown"
    t = str(tok).strip().lower()
    # clean separators, allow combined labels
    parts = re.split(r"[^\w]+", t)
    parts = [p for p in parts if p]
    parts_set = set(parts)
    if "blink" in parts_set or "blinking" in parts_set or "blinked" in parts_set:
        return "blink"
    if "center" in parts_set or "centre" in parts_set:
        return "center"
    if "left" in parts_set:
        return "left"
    if "right" in parts_set:
        return "right"
    return "unknown"

def eye_debug(tokens: pd.Series) -> dict:
    if tokens is None or tokens.dropna().empty:
        return {"eye_center_frac_5s": np.nan, "eye_blink_frac_5s": np.nan, "eye_lr_frac_5s": np.nan, "eye_total_tokens_5s": 0}
    norm = tokens.dropna().map(_normalize_gaze_token)
    total = len(norm)
    counts = Counter(norm)
    return {
        "eye_center_frac_5s": (counts.get("center",0)/total),
        "eye_blink_frac_5s":  (counts.get("blink",0)/total),
        "eye_lr_frac_5s":     ((counts.get("left",0)+counts.get("right",0))/total),
        "eye_total_tokens_5s": total,
    }

def interpret_eye_tracking(tokens: pd.Series,
                           center_thr=0.50,
                           blink_thr=0.20,
                           lr_thr=0.50) -> str:
    if tokens is None or tokens.dropna().empty:
        return "unknown"
    norm = tokens.dropna().map(_normalize_gaze_token)
    total = len(norm)
    counts = Counter(norm)
    center_frac = counts.get("center", 0) / total
    blink_frac  = counts.get("blink", 0) / total
    lr_frac     = (counts.get("left", 0) + counts.get("right", 0)) / total
    # Priority: blinks > focused > distracted; fallback plurality
    if blink_frac >= blink_thr:
        return "blinking_frequently"
    if center_frac >= center_thr:
        return "focused"
    if lr_frac >= lr_thr:
        return "distracted"
    label = counts.most_common(1)[0][0]
    return {"center":"focused", "blink":"blinking_frequently"}.get(label, "distracted")

# ------------------------ Group Interpreters ------------------------

def interpret_room_noise(df_chunk: pd.DataFrame) -> str:
    c_noise = None
    for k in ["noise_level","noiselevel","noise_db","db","sound_level"]:
        c_noise = find_col(df_chunk, k)
        if c_noise: break
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
    c_cat = find_col(df_chunk, ["mouse_movement","mouse movement"])
    if c_cat:
        vals = [str(x).lower() for x in df_chunk[c_cat].dropna()]
        if not vals: return "unknown"
        frac_active = sum(("move" in v or "active" in v) for v in vals)/len(vals)
        return "active" if frac_active > 0.30 else "calm"
    c_speed = find_col(df_chunk, ["Speed","mouse_speed","cursor_speed","pointer_speed"])
    if c_speed:
        v = pd.to_numeric(df_chunk[c_speed], errors="coerce").mean()
        if pd.isna(v): return "unknown"
        return "active" if v >= 5.0 else "calm"
    return "unknown"

def interpret_body_posture(df_chunk: pd.DataFrame) -> str:
    c_posture = find_col(df_chunk, ["posture","body_seating_posture"])
    c_head    = find_col(df_chunk, "head_forward")
    c_streak  = find_col(df_chunk, "bad_posture_streak")
    head_flag = None
    if c_head:
        head_flag = (pd.to_numeric(df_chunk[c_head], errors="coerce").fillna(0).mean() >= 0.5)
    streak_bad = None
    if c_streak:
        streak_bad = (pd.to_numeric(df_chunk[c_streak], errors="coerce").fillna(0).mean() >= 2)
    posture_label = None
    if c_posture:
        posture_label = mode_or_unknown(df_chunk[c_posture])
        if "good" in posture_label.lower(): posture_label = "good"
        elif any(k in posture_label.lower() for k in ["bad","poor","slouch"]): posture_label = "poor"
        else: posture_label = "unknown"
    if head_flag:
        return "head_forward"
    if posture_label == "poor" or (streak_bad is True):
        return "poor"
    if posture_label == "good":
        return "good"
    return "unknown"

def interpret_body_movement(df_chunk: pd.DataFrame) -> str:
    c_status = find_col(df_chunk, "movement_status")
    c_depth  = find_col(df_chunk, "depth_change")
    status = None
    if c_status:
        vals = [str(x).lower() for x in df_chunk[c_status].dropna()]
        if vals:
            frac_move = sum(("move" in v or "active" in v or "walking" in v or "standing" in v) for v in vals)/len(vals)
            status = "moving" if frac_move > 0.30 else "still"
    if c_depth:
        v = pd.to_numeric(df_chunk[c_depth], errors="coerce").abs().mean()
        if pd.notna(v) and v > 0.5:
            return "moving"
    return status if status else "still"

def interpret_voice(df_chunk: pd.DataFrame) -> str:
    c_state = find_col(df_chunk, ["voice_state","voice state","voice_detection","voice"])
    c_wpm   = find_col(df_chunk, ["WPM","wpm"])
    if c_state:
        vals = [str(x).lower() for x in df_chunk[c_state].dropna()]
        if any(("talk" in v or "speak" in v or "active" in v) for v in vals): return "talking"
        if any(("silent" in v or "none" in v) for v in vals): return "silent"
    if c_wpm:
        v = pd.to_numeric(df_chunk[c_wpm], errors="coerce").mean()
        if pd.notna(v) and v >= 10: return "talking"
    return "silent"

# ------------------------ Pipeline ------------------------

def preprocess_to_5s_groupaware_v3(input_csv: str, output_csv: str, smooth_eye=True) -> pd.DataFrame:
    df = pd.read_csv(input_csv)

    # Timestamp
    ts_col = find_timestamp_col(df)
    if ts_col is None:
        raise ValueError("No timestamp-like column found.")
    df["timestamp"] = pd.to_datetime(df[ts_col], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    df["t_5s"] = df["timestamp"].dt.floor("5S")

    # Identify canonical environment/vitals
    def pick(key_list):
        c = find_col(df, key_list)
        return c if c in df.columns else None

    env_map = {
        "room_temperature": pick(["temperature_C","room_temperature","temperature"]),
        "room_pressure": pick(["pressure_hPa","pressure","baro"]),
        "room_humidity": pick(["humidity_%","humidity","rh_%"]),
        "room_light_intensity": pick(["light_lux","lux","room_light_intensity"]),
        "skin_temperature": pick(["skin_temperature_C","skin_temp"]),
        "light_temperature": pick(["CCT_K","cct_k","light_temperature"]),
        "heart_rate": pick(["Heart Rate (bpm)","heart_rate_bpm","hr_bpm"]),
    }

    # Aggregate numeric means
    keep_numeric = [c for c in env_map.values() if c]
    grouped = df[["t_5s"] + keep_numeric].groupby("t_5s").mean().reset_index()
    # Rename to parent names
    for parent, src in env_map.items():
        if src and src in grouped.columns:
            grouped.rename(columns={src: parent}, inplace=True)

    # Eye tracking interpretation + debug
    col_gaze = pick(["eye_tracking","eye tracking","eye_gaze","gaze","eye direction","eye movement","gaze token"])
    if col_gaze:
        eye_labels = df.groupby("t_5s")[col_gaze].apply(interpret_eye_tracking).reset_index(name="eye_tracking")
        grouped = grouped.merge(eye_labels, on="t_5s", how="left")
        # Debug fractions
        dbg = df.groupby("t_5s")[col_gaze].apply(eye_debug).reset_index()
        dbg = pd.concat([dbg["t_5s"], dbg[col_gaze].apply(pd.Series)], axis=1)
        grouped = grouped.merge(dbg, on="t_5s", how="left")
    else:
        grouped["eye_tracking"] = "unknown"
        grouped["eye_center_frac_5s"] = np.nan
        grouped["eye_blink_frac_5s"] = np.nan
        grouped["eye_lr_frac_5s"] = np.nan
        grouped["eye_total_tokens_5s"] = 0

    # Room noise
    noise_labels = df.groupby("t_5s").apply(interpret_room_noise).reset_index(name="room_noise")
    grouped = grouped.merge(noise_labels, on="t_5s", how="left")

    # Mouse movement
    mouse_labels = df.groupby("t_5s").apply(interpret_mouse).reset_index(name="mouse_movement")
    grouped = grouped.merge(mouse_labels, on="t_5s", how="left")

    # Body posture
    posture_labels = df.groupby("t_5s").apply(interpret_body_posture).reset_index(name="body_posture")
    grouped = grouped.merge(posture_labels, on="t_5s", how="left")

    # Body movement
    movement_labels = df.groupby("t_5s").apply(interpret_body_movement).reset_index(name="body_movement")
    grouped = grouped.merge(movement_labels, on="t_5s", how="left")

    # Voice
    voice_labels = df.groupby("t_5s").apply(interpret_voice).reset_index(name="voice")
    grouped = grouped.merge(voice_labels, on="t_5s", how="left")

    # Face emotion / stress / task timestamp (modes)
    col_emotion = pick(["Emotion Detection","emotion_detection","emotion"])
    if col_emotion:
        emo = df.groupby("t_5s")[col_emotion].apply(mode_or_unknown).reset_index(name="face_emotion")
        grouped = grouped.merge(emo, on="t_5s", how="left")
    else:
        grouped["face_emotion"] = "unknown"

    col_stress = pick(["face_stress","stress","stress_label","stress probability","stress_probability"])
    if col_stress:
        st = df.groupby("t_5s")[col_stress].apply(mode_or_unknown).reset_index(name="face_stress")
        grouped = grouped.merge(st, on="t_5s", how="left")
    else:
        grouped["face_stress"] = "unknown"

    col_task = pick(["task timestamp","current task timestamp","Task Timestamp","Current Task Timestamp"])
    if col_task:
        tt = df.groupby("t_5s")[col_task].apply(mode_or_unknown).reset_index(name="task_timestamp")
        grouped = grouped.merge(tt, on="t_5s", how="left")
    else:
        grouped["task_timestamp"] = ""

    # Optional smoothing for eye_tracking (mode over 3 bins)
    if smooth_eye and "eye_tracking" in grouped.columns:
        def smooth_mode(series, k=3):
            out = []
            for i in range(len(series)):
                window = series[max(0, i - k + 1): i + 1]
                m = window.mode()
                out.append(m.iat[0] if not m.empty else window.iloc[-1])
            return pd.Series(out, index=series.index, dtype=object)
        grouped["eye_tracking_smooth"] = smooth_mode(grouped["eye_tracking"], k=3)

    # Finalize timestamp + order columns
    grouped.rename(columns={"t_5s":"timestamp"}, inplace=True)
    order_front = [
        "timestamp",
        "eye_tracking","eye_tracking_smooth","eye_center_frac_5s","eye_blink_frac_5s","eye_lr_frac_5s","eye_total_tokens_5s",
        "mouse_movement","body_posture","body_movement","voice",
        "face_emotion","face_stress","task_timestamp","room_noise",
        "room_temperature","room_pressure","room_humidity","room_light_intensity",
        "skin_temperature","light_temperature","heart_rate",
    ]
    order_front += [c for c in grouped.columns if c not in order_front]
    grouped = grouped.loc[:, order_front]

    grouped.to_csv(output_csv, index=False)
    return grouped

# ------------------------ CLI ------------------------

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="HPMS 5s preprocessor (v3) with robust eye-tracking and debug.")
    ap.add_argument("input_csv", help="Path to raw input CSV")
    ap.add_argument("output_csv", help="Path to output CSV (5-second interpreted)")
    ap.add_argument("--no-smooth", action="store_true", help="Disable eye_tracking smoothing")
    args = ap.parse_args()
    out = preprocess_to_5s_groupaware_v3(args.input_csv, args.output_csv, smooth_eye=(not args.no_smooth))
    print(f"Wrote {args.output_csv} with {len(out)} rows and {len(out.columns)} columns.")
