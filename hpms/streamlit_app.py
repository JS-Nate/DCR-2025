# streamlit_app.py
# ---------------------------------------------------------------------
# Two-page UI:
#   - HPMS Dashboard: main pipeline (upload CSV, run analysis, LLaMA physio feedback, suggestions)
#   - HPSN Explorer: inspect Module 5 (graph, nodes/edges, explanations, live reasoning)
# ---------------------------------------------------------------------

import datetime
from datetime import datetime as dt, timedelta
import json
from typing import Any
import time
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

# ---------- Helpers for LLaMA render ----------
def _extract_json_and_narrative(txt: str):
    """Return (json_obj, narrative_str) or (None, cleaned_txt). Robustly finds first JSON block."""
    if not isinstance(txt, str) or "{" not in txt:
        return None, (txt or "").strip()
    s = txt.find("{")
    depth, i, in_str, esc = 0, s, False, False
    while i < len(txt):
        ch = txt[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    json_part = txt[s:i+1]
                    rest = txt[i+1:].strip()
                    import json as _json
                    try:
                        obj = _json.loads(json_part)
                        return obj, rest
                    except Exception:
                        break
        i += 1
    return None, txt.strip()

def _chip(label, color="gray"):
    colors = {
        "green": "#16a34a", "amber": "#d97706", "red": "#dc2626",
        "blue": "#2563eb", "gray": "#6b7280"
    }
    return f"<span style='display:inline-block;padding:2px 8px;border-radius:12px;background:{colors.get(color,'#6b7280')};color:#fff;font-size:12px;margin-right:6px'>{label}</span>"

def render_physio_feedback_ui(data: dict, narrative: str = ""):
    st.subheader("Structured Physiological Feedback")

    # Summary chips
    summary = (data or {}).get("summary", {})
    state = str(summary.get("state", "unknown")).lower()
    color = "green" if state == "optimal" else "amber" if state == "watch" else "red" if state == "critical" else "gray"
    drivers = summary.get("drivers", []) or []
    conf = summary.get("confidence", None)

    chips = _chip(f"State: {state.upper()}", color)
    if conf is not None:
        try:
            pct = round(float(conf) * 100)
            chips += " " + _chip(f"Confidence: {pct:d}%", "blue")
        except Exception:
            pass
    if drivers:
        chips += " " + "".join(_chip(f"Driver: {d}", "gray") for d in drivers[:3])
    st.markdown(chips, unsafe_allow_html=True)

    # Observations table
    obs = data.get("observations", [])
    if obs:
        df_obs = pd.DataFrame(obs)
        st.markdown("**Observations**")
        st.dataframe(df_obs, use_container_width=True)

    # Interpretation bullets
    interp = data.get("physiological_interpretation", [])
    if interp:
        st.markdown("**Interpretation**")
        for b in interp:
            st.markdown(f"- {b}")

    # Recommendations
    recs = data.get("physiological_health_recommendations", [])
    if recs:
        st.markdown("**Physiological Health Recommendations**")
        for r in recs:
            with st.container(border=True):
                st.markdown(f"**{r.get('label','')}** — {r.get('why','')}")
                st.markdown(f"- *How*: {r.get('how_to','')}")
                st.markdown(f"- *Duration*: {r.get('duration_s','?')}s")
                st.markdown(f"- *Expected effect*: {r.get('expected_effect','')}")
                if r.get("monitor"):
                    st.markdown(f"- *Monitor*: {r['monitor']}")
                if r.get("constraints"):
                    st.caption(f"Constraints: {r['constraints']}")

    if narrative:
        st.markdown("**Brief Narrative**")
        st.write(narrative[:800])


def show_df(df, *, use_container_width=True):
    """Try to use st.dataframe (fast, scrollable). If PyArrow is missing, fallback to st.table."""
    try:
        st.dataframe(df, use_container_width=use_container_width)
    except Exception as e:
        st.warning(f"Falling back to a static table (PyArrow missing): {e}")
        st.table(df)


# ---------- Optional LLaMA utils (kept on HPMS page) ----------
try:
    import llama_utils  # keeps your original LLaMA analysis section working
    HAS_LLAMA = True
except Exception:
    HAS_LLAMA = False

# ---------- Analysis (existing functions) ----------
from analysis import (
    compute_psf_loads,        # per-group loads + input load
    compute_output_score,     # output-side score (0–1)
    compute_performance,      # combines input/output into a performance score/state
    check_safety_margins,     # tabular safety margin status
    generate_environment_commands,  # legacy environment suggestions
    build_local_physiological_feedback # local fallback for operator coaching
)

# ---------- HPSN (Module 5) ----------
from hpsn import (
    load_hpsn, query_hpsn, update_hpsn, map_inputs_to_nodes, save_hpsn,
    get_config_bundle, get_measure_catalog, get_measure_mapping, get_threshold_profiles, get_estimation_spec,
    infer_state, predict_state, explain,
    get_version_info, get_structure_report, list_nodes, list_edges, get_node_details,
    get_recent_explanations, get_graph_dot, get_ego_graph_dot,
)


# ================== App Setup ==================
st.set_page_config(page_title="SMR HPMS + HPSN", layout="wide")
load_hpsn()

# Sidebar navigation + options
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["HPMS Dashboard", "HPSN Explorer"], index=0)

st.sidebar.markdown("### Options")
use_llama = st.sidebar.checkbox("Enable LLaMA Physiological Feedback", value=True)

# Streaming controls
st.sidebar.markdown("### Streaming")
simulate_stream = st.sidebar.checkbox("Simulate 1‑min streaming", value=False)
stream_speed = st.sidebar.slider("Seconds to display each 1‑min window", 1, 60, 60)
force_update = st.sidebar.button('Update page', help='Manually refresh the analysis now')


# Session state
if "df" not in st.session_state:
    st.session_state.df = None
if "last_row" not in st.session_state:
    st.session_state.last_row = None
if "alert_log" not in st.session_state:
    st.session_state.alert_log = []
if "feedback_log" not in st.session_state:
    st.session_state.feedback_log = []
if "stream_windows" not in st.session_state:
    st.session_state.stream_windows = []
if "stream_idx" not in st.session_state:
    st.session_state.stream_idx = 0
if "stream_running" not in st.session_state:
    st.session_state.stream_running = False

# --- NEW: streaming-by-rows settings/state ---
if "stream_chunk_bounds" not in st.session_state:
    st.session_state.stream_chunk_bounds = []   # list of (start_idx, end_idx_exclusive)
if "stream_chunk_size" not in st.session_state:
    st.session_state.stream_chunk_size = 12     # 12 rows == 1 minute window
if "stream_window_s" not in st.session_state:
    st.session_state.stream_window_s = 60       # display each window for 60s
if "next_switch_at" not in st.session_state:
    st.session_state.next_switch_at = None      # datetime to advance to next chunk


# ================== Helpers ==================
def safe_str(x, default="unknown"):
    try:
        s = str(x).strip()
        return s if s else default
    except Exception:
        return default

def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default

def safe_int(x, default=0):
    try:
        return int(float(x))
    except Exception:
        return default

def pretty_json(obj: Any):
    st.code(json.dumps(obj, indent=2, ensure_ascii=False, default=str), language="json")


# =================================================================
# Page 1: HPMS Dashboard (keeps your analysis here)
# =================================================================
if page == "HPMS Dashboard":
    st.title("SMR Human Performance Management System (HPMS)")
    st.caption("Real-time Output/Input performance with PSF loads, alerts, and HPSN integration.")

    # ---- Data upload ----
    with st.expander("Upload Scenario Data (CSV)", expanded=True):
        uploaded_file = st.file_uploader("Upload CSV file", type="csv")
        if uploaded_file:
            df = pd.read_csv(uploaded_file)

            # --- Adapter for known column names ---
            rename_map = {
                "skin_temperature": "skin_temp",
                "body_posture": "posture",
                "room_temperature": "room_temp",
                "light_temperature": "cct_temp",
                "room_light_intensity": "light_intensity",
                "room_humidity": "humidity",
                "room_pressure": "pressure",
            }
            df = df.rename(columns={c: rename_map.get(c, c) for c in df.columns})

            # Parse and clean timestamp
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

            # Derive face_stress if not present
            if "face_stress" not in df.columns:
                def _derive_stress(em):
                    e = str(em).strip().lower()
                    if e in ("sad","fear","angry"): return "high"
                    if e in ("surprised",): return "medium"
                    return "low"
                df["face_stress"] = df.get("face_emotion", "").map(_derive_stress)

            # Task column
            task_col = "task_description" if "task_description" in df.columns else ("task_timestamp" if "task_timestamp" in df.columns else None)
            if task_col:
                df["task"] = df[task_col].fillna("unknown")
            elif "task" not in df.columns:
                df["task"] = "unknown"

            # Compute per-row task_start and task_duration
            df["task_start"] = None
            current_start, current_task = None, "unknown"
            if task_col:
                for i, row_i in df.iterrows():
                    if isinstance(row_i[task_col], str) and row_i[task_col].strip():
                        current_start = row_i["timestamp"]
                        current_task = row_i[task_col]
                    df.at[i, "task_start"] = current_start
                    if not df.at[i, "task"]:
                        df.at[i, "task"] = current_task
            df["task_start"] = pd.to_datetime(df["task_start"], errors="coerce")
            df["task_duration"] = (df["timestamp"] - df["task_start"]).dt.total_seconds().fillna(0).astype(int)

            # Reactor status default if missing
            if "reactor_status" not in df.columns:
                df["reactor_status"] = "normal"

            # Required inputs
            required_cols = [
                'timestamp', 'heart_rate', 'skin_temp', 'posture', 'eye_tracking', 'voice',
                'face_emotion', 'face_stress', 'task', 'task_duration',
                'room_temp', 'cct_temp', 'light_intensity', 'humidity', 'pressure'
            ]
            missing = [c for c in required_cols if c not in df.columns]
            if missing:
                st.error(f"CSV is missing required columns: {missing}")
            else:
                # Optional outputs & reactor flag
                optional_cols = [
                    'task_success', 'completion_time_s', 'decision_accuracy',
                    'operational_stability', 'situational_awareness', 'reactor_status'
                ]
                for c in optional_cols:
                    if c not in df.columns:
                        df[c] = None

                # Prepare time and sort
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                df = df.dropna(subset=['timestamp']).sort_values(by='timestamp').reset_index(drop=True)
                st.session_state.df = df
                st.success(f"Loaded {len(df)} rows.")
                show_df(df.head())

    df = st.session_state.df

    if df is not None and not df.empty:
        # ----------------- STREAMING (12 rows / 60s) -----------------
        # Build chunk bounds once (or when df length changes)
        if (not st.session_state.stream_chunk_bounds) or \
           (st.session_state.stream_chunk_bounds and st.session_state.stream_chunk_bounds[-1][1] != len(df)):
            st.session_state.stream_chunk_bounds = [(i, min(i + st.session_state.stream_chunk_size, len(df)))
                                                    for i in range(0, len(df), st.session_state.stream_chunk_size)]
            st.session_state.stream_idx = 0
            st.session_state.next_switch_at = None


        # Advance to next window if deadline passed (triggered on rerun from the component)
        if simulate_stream and st.session_state.next_switch_at is not None and dt.utcnow() >= st.session_state.next_switch_at:
            st.session_state.stream_idx += 1
            if st.session_state.stream_idx >= len(st.session_state.stream_chunk_bounds):
                st.session_state.stream_idx = len(st.session_state.stream_chunk_bounds) - 1  # clamp last
            st.session_state.next_switch_at = dt.utcnow() + timedelta(seconds=st.session_state.stream_window_s)
        # Tie slider to window seconds (user can override 60s)
        if stream_speed and stream_speed != st.session_state.stream_window_s:
            st.session_state.stream_window_s = int(stream_speed)

        # Streaming heartbeat
        if simulate_stream:
            # Initialize deadline
            if st.session_state.next_switch_at is None:
                st.session_state.next_switch_at = dt.utcnow() + timedelta(seconds=st.session_state.stream_window_s)

            # Countdown and auto-switch
            secs_left = max(0, int((st.session_state.next_switch_at - dt.utcnow()).total_seconds()))
            if secs_left <= 0:
                st.session_state.stream_idx += 1
                if st.session_state.stream_idx >= len(st.session_state.stream_chunk_bounds):
                    st.session_state.stream_idx = len(st.session_state.stream_chunk_bounds) - 1  # clamp last
                st.session_state.next_switch_at = dt.utcnow() + timedelta(seconds=st.session_state.stream_window_s)
                # Force a re-run to show the next window immediately
                st.experimental_set_query_params(_=dt.utcnow().strftime("%H%M%S"))

        # Pick the current chunk (either streaming or manual)
        cur_chunk_idx = st.session_state.stream_idx if simulate_stream else st.number_input(
            "Manual window (12-row chunk #)",
            min_value=0,
            max_value=max(0, len(st.session_state.stream_chunk_bounds)-1),
            value=min(st.session_state.stream_idx, max(0, len(st.session_state.stream_chunk_bounds)-1)),
            step=1
        )

        if not st.session_state.stream_chunk_bounds:
            st.warning("No data windows available.")
            st.stop()

        start_i, end_i = st.session_state.stream_chunk_bounds[cur_chunk_idx]
        window_df = df.iloc[start_i:end_i].copy()

        # Show window context
        left_info, right_info = st.columns(2)
        with left_info:
            st.subheader("Current 1-Minute Window")
            st.caption(f"Rows {start_i}–{end_i-1} of {len(df)-1}  •  Window #{cur_chunk_idx+1}/{len(st.session_state.stream_chunk_bounds)}")
            show_df(window_df)
        with right_info:
            if simulate_stream:
                secs_left = max(0, int((st.session_state.next_switch_at - dt.utcnow()).total_seconds()))
                st.metric("Time left in window", f"{secs_left}s")
                total = max(1, st.session_state.stream_window_s)
                st.progress(min(1.0, 1.0 - (secs_left / total)))
                # Schedule a no-reload, one-shot rerun at the deadline using a Streamlit component
                ms_left = max(0, int((st.session_state.next_switch_at - dt.utcnow()).total_seconds() * 1000))
                # small safety buffer
                ms_left = ms_left + 100
                _ = components.html(
                    f"""
                    <script>
                      var ms = {ms_left};
                    </script>
                    <script src="https://unpkg.com/streamlit-component-lib/dist/index.js"></script>
                    <script>
                      // Notify Streamlit (no page reload). This updates the component's value and triggers a rerun.
                      function triggerRerun() {{ Streamlit.setComponentValue(Date.now()); }}
                      // If ms is 0, send immediately; otherwise schedule.
                      if (ms <= 0) {{ triggerRerun(); }} else {{ setTimeout(triggerRerun, ms); }}
                    </script>
                    """,
                    height=0
                )




        # Choose the last row of the window for downstream analysis (mimics "latest reading" in that minute)
        row = window_df.iloc[-1]
        st.session_state.last_row = row.to_dict()

        # ============== ORIGINAL ANALYSIS USING `row` ==============
        st.subheader("Run Analysis on Current Window (last reading)")

        # -------- Extract inputs --------
        hr = safe_float(row.get('heart_rate'), 0)
        skin_temp = safe_float(row.get('skin_temp'), 0)
        posture = safe_str(row.get('posture'), 'unknown')
        eye_tracking = safe_str(row.get('eye_tracking'), 'unknown')
        voice = safe_str(row.get('voice'), 'unknown')
        task = safe_str(row.get('task'), 'unknown')
        task_duration = safe_int(row.get('task_duration'), 0)

        emotion = safe_str(row.get('face_emotion'), 'neutral').lower()
        stress = safe_str(row.get('face_stress'), 'low').lower()

        temp = safe_float(row.get('room_temp'), 0)
        cct_temp = safe_float(row.get('cct_temp'), 0)
        light_intensity = safe_float(row.get('light_intensity'), 0)
        humidity = safe_float(row.get('humidity'), 0)
        pressure = safe_float(row.get('pressure'), 0)

        reactor_status = safe_str(row.get('reactor_status'), 'normal').lower()

        # -------- Optional outputs --------
        task_success = row.get('task_success', None)
        completion_time_s = row.get('completion_time_s', None)
        decision_accuracy = row.get('decision_accuracy', None)
        operational_stability = row.get('operational_stability', None)
        situational_awareness = row.get('situational_awareness', None)

        # -------- HPSN (legacy suggestions) --------
        input_nodes = map_inputs_to_nodes(row)
        hpsn_suggestions = query_hpsn(input_nodes)
        hpsn_suggestions_text = "\n".join([f"- {action} (weight: {weight:.2f})" for action, weight in hpsn_suggestions]) if hpsn_suggestions else "—"

        # -------- Safety margins (fixed arg order) --------
        safety_data = check_safety_margins(
            hr, skin_temp, temp, emotion, stress, cct_temp, light_intensity, humidity, pressure
        )

        # -------- PSF Loads → Input Load --------
        per_group_loads, input_load_score = compute_psf_loads(
            hr, skin_temp, posture, eye_tracking, voice,
            emotion, stress, temp, cct_temp, light_intensity, humidity, pressure,
            task, task_duration, reactor_status
        )

        # -------- Output Score --------
        output_score = compute_output_score(
            task_success=task_success,
            completion_time_s=completion_time_s,
            decision_accuracy=decision_accuracy,
            operational_stability=operational_stability,
            situational_awareness=situational_awareness,
            fallback_task_duration=task_duration
        )

        # -------- Performance --------
        perf_score, perf_state = compute_performance(per_group_loads, input_load_score, output_score)

        # ----------------- TOP KPIs -----------------
        c1, c2, c3 = st.columns(3)
        c1.metric("Performance Score (Output / Input)", f"{perf_score:.3f}", help="Higher = more efficient output given the current load.")
        c2.metric("Output Score (0–1)", f"{output_score:.2f}")
        c3.metric("Input Load (0–1)", f"{input_load_score:.2f}")

        # State banner
        if perf_state == "normal":
            st.success("State: NORMAL – Performance within expected range.")
        elif perf_state == "warning":
            st.warning("State: WARNING – Performance trending down. Review load drivers.")
        else:
            st.error("State: CRITICAL – Performance degraded. Immediate attention required.")

        # ----------------- DRIVERS TABLE -----------------
        st.markdown("### PSF Load Breakdown (by category)")
        drivers_df = pd.DataFrame.from_dict(per_group_loads, orient="index", columns=["load"]).sort_values("load", ascending=False)
        show_df(drivers_df.round(3))

        # Log alert entries for recent display
        top_driver = drivers_df.index[0] if not drivers_df.empty else "unknown"
        if perf_state in ["warning", "critical"]:
            st.session_state.alert_log.append({
                "timestamp": row['timestamp'],
                "state": perf_state,
                "perf_score": round(perf_score, 3),
                "output_score": round(output_score, 2),
                "input_load": round(input_load_score, 2),
                "top_driver": safe_str(top_driver)
            })

        # ----------------- PERFORMANCE HIERARCHY (Safety Margins) -----------------
        st.markdown("### Performance Hierarchy Status (Safety Margins)")
        show_df(pd.DataFrame(safety_data, columns=["ID","Parameter","Value","Status"]).astype(str))

        # ----------------- RECENT ALERT LOG -----------------
        st.markdown("### Recent Alert Log")
        if st.session_state.alert_log:
            df_log = pd.DataFrame(st.session_state.alert_log[-10:])
            show_df(df_log.astype(str))
        else:
            st.success("No recent performance alerts.")

        # ----------------- HPSN SUGGESTIONS (legacy) -----------------
        st.markdown("### HPSN Suggestions (legacy query_hpsn)")
        st.write(hpsn_suggestions_text if hpsn_suggestions else "No HPSN suggestions available.")

        # ----------------- Module 5 — Live Reasoning Output -----------------
        st.markdown("### HPSN — Live Reasoning (infer_state / predict_state)")
        payload = {
            "version": "v1",
            "timestamp": (
                row.get("timestamp").isoformat()
                if isinstance(row.get("timestamp"), (datetime.datetime, pd.Timestamp, pd.NaT.__class__))
                else datetime.datetime.utcnow().isoformat()
            ),
            "operator_id": "opA",
            "task": {"id": task, "duration_s": task_duration, "mode": reactor_status},
            "signals": {
                "hr_bpm": hr,
                "skin_temp": skin_temp,
                "posture": posture,
                "eye_tracking": eye_tracking,
                "voice": voice,
                "emotion": emotion,
                "stress": stress,
                "room_temp": temp,
                "cct_temp_k": cct_temp,
                "light_intensity": light_intensity,
                "humidity": humidity,
                "pressure": pressure,
            },
            "reactor_status": reactor_status,
        }
        colA, colB = st.columns(2)
        with colA:
            st.caption("infer_state()")
            inf = infer_state(payload)
            pretty_json(inf)
        with colB:
            st.caption("predict_state()")
            horizon = st.slider("Forecast horizon (seconds)", 60, 3600, 600, 60)
            pred = predict_state(payload, horizon_s=horizon)
            pretty_json(pred)

        # Explanation retrieval
        exp_id = inf.get("explanation_id") if isinstance(inf, dict) else None
        if exp_id:
            with st.expander("Show explanation trace"):
                pretty_json(explain(exp_id))

        # ----------------- 🫀 Physiological Feedback (only when enabled) -----------------
        if use_llama:
            st.markdown("### 🫀 Physiological Feedback")
            row_json = json.dumps(st.session_state.last_row or {}, default=str)
            hpsn_evidence = {
                "per_group_loads": per_group_loads if isinstance(per_group_loads, dict) else {},
                "input_load": input_load_score if isinstance(input_load_score, (int, float)) else None,
            }
            hpsn_json = json.dumps(hpsn_evidence, default=str)

            physio_prompt = f"""
You are a safety-aware assistant producing physiological feedback for a nuclear control room operator.
Focus ONLY on human physiology and operator self-regulation. Do NOT make plant/environment control decisions.

Limits (must follow exactly):
- Heart rate: 60–100 bpm (normal)
- Skin temperature: 32.0–36.0 °C (normal)
- Face stress: safe=['low'], warning=['medium'], danger=['high']
- Face emotion: safe=['happy','neutral'], warning=['surprised'], danger=['sad','fear','angry']

HPSN evidence (optional):
{hpsn_json}

RAW ROW (single snapshot):
{row_json}

TASK:
1) Return ONE JSON object with EXACT keys:

{{
  "summary": {{"state":"optimal|watch|critical","drivers":[],"confidence":0.0}},
  "observations": [
    {{"param":"hr_bpm","value":<num>,"status":"below|normal|above","reason":"..."}},
    {{"param":"skin_temp","value":<num>,"status":"below|normal|above","reason":"..."}},
    {{"param":"face_stress","value":"low|medium|high","status":"safe|warning|danger","reason":"..."}},
    {{"param":"emotion","value":"...","status":"safe|warning|danger","reason":"..."}}
  ],
  "physiological_interpretation": [],
  "physiological_health_recommendations": [
    {{
      "label":"paced_breathing_60s",
      "why":"e.g., hr_bpm high OR stress=medium/high",
      "how_to":"Inhale 4s, exhale 6s for 60s, seated, shoulders relaxed.",
      "duration_s":60,
      "expected_effect":"lower sympathetic arousal; HR trend down",
      "monitor":"recheck HR and stress label after 5–10 min",
      "constraints":"do not perform if feeling dizzy; pause and notify supervisor"
    }}
  ],
  "monitoring_next":[
    "Watch HR and facial-stress label for 5–10 min after intervention"
  ]
}}

2) After the JSON, write <=100 words summarizing state and the top 1–2 recommended steps.
Return valid JSON FIRST, then the narrative.
"""
            left, right = st.columns(2)

            raw = ""
            with left:
                if HAS_LLAMA:
                    with st.expander("Run LLaMA Physiological Feedback"):
                        model = st.text_input("Ollama model", value="llama2")
                        try:
                            raw = llama_utils.query_llama(physio_prompt)
                        except Exception as e:
                            raw = f"[ollama error] {e}"
                        st.text_area("Raw LLaMA Output", raw, height=240)
                else:
                    st.caption("llama_utils.py not loaded.")

            with right:
                import json as _json
                data, narrative = (None, "")
                if HAS_LLAMA and raw:
                    data, narrative = _extract_json_and_narrative(raw)
                if data:
                    with st.expander("Show JSON (debug)"):
                        st.code(_json.dumps(data, indent=2), language="json")
                    render_physio_feedback_ui(data, narrative)
                else:
                    # Local fallback
                    fallback = {
                        "summary": {"state": "watch", "drivers": ["physiological"], "confidence": 0.7},
                        "observations": [],
                        "physiological_interpretation": [],
                        "physiological_health_recommendations": build_local_physiological_feedback(
                            hr=hr,
                            skin_temp=skin_temp,
                            stress=stress,
                            emotion=emotion,
                            posture=posture,
                            eye_tracking=eye_tracking,
                            voice=voice,
                            humidity=humidity,
                        ),
                        "monitoring_next": ["Watch HR and facial-stress label for 5–10 min after intervention"],
                    }
                    render_physio_feedback_ui(fallback, "Fallback guidance shown. Apply the first recommendation and reassess in 5–10 minutes.")
        # else: fully hidden when disabled

        # ----------------- REPORT -----------------
        st.markdown("### Scenario Summary Report")
        report_text = f"""

Scenario Report: {task}
Timestamp: {row['timestamp']}

Inputs:
- Heart Rate: {hr} bpm
- Skin Temperature: {skin_temp} °C
- Posture: {posture}
- Eye Tracking: {eye_tracking}
- Voice: {voice}
- Emotion: {emotion}
- Facial Stress: {stress}
- Room Temp: {temp} °C
- CCT Temp: {cct_temp} K
- Light Intensity: {light_intensity} lux
- Humidity: {humidity} %
- Pressure: {pressure} hPa
- Reactor Status: {reactor_status}

Outputs (if provided / inferred):
- Task Success: {task_success}
- Completion Time (s): {completion_time_s if completion_time_s is not None else task_duration}
- Decision Accuracy: {decision_accuracy}
- Operational Stability: {operational_stability}
- Situational Awareness: {situational_awareness}

Scores:
- Output Score: {output_score:.2f}
- Input Load: {input_load_score:.2f}
- Performance (Output/Input): {perf_score:.3f} → {perf_state.upper()}
- Top PSF Driver: {top_driver}

HPSN Suggestions:
{hpsn_suggestions_text}
"""
        st.text_area("Generated Report:", report_text, height=320)
        st.download_button(
            label="Download Scenario Report",
            data=str(report_text),
            # file_name=f"scenario_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            file_name = f"scenario_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

        )

        # ----------------- ENVIRONMENT COMMANDS -----------------
        st.markdown("### Environment Adjustment Commands")
        env_cmds = generate_environment_commands(cct_temp, light_intensity, humidity, pressure)
        if env_cmds:
            for cmd in env_cmds:
                st.code(cmd, language="python")
        else:
            st.caption("Environment within nominal ranges. No commands generated.")
    
        # ---- Auto-refresh (1s) placed at END so all panels render before rerun ----
        if simulate_stream:
            time.sleep(1)
            try:
                st.rerun()
            except Exception:
                try:
                    st.experimental_rerun()
                except Exception:
                    pass

        else:
                st.info("Upload a CSV to begin. Required columns are:")
                st.code(", ".join([
                    'timestamp','heart_rate','skin_temp','posture','eye_tracking','voice',
                    'face_emotion','face_stress','task','task_duration',
                    'room_temp','cct_temp','light_intensity','humidity','pressure'
                ]))
                st.caption("Optional outputs: task_success, completion_time_s, decision_accuracy, operational_stability, situational_awareness, reactor_status.")

# =================================================================
# Page 2: HPSN Explorer
# =================================================================
elif page == "HPSN Explorer":

    st.title("HPSN Explorer (Interactive & Hierarchical)")

    # ------------- TOP-LEVEL OVERVIEW -------------
    with st.container(border=True):
        rep = get_structure_report(top_k=10)
        summary = rep["summary"]
        node_type_counts = rep["node_type_counts"]
        measure_count = rep["measure_count"]
        colA, colB, colC, colD = st.columns(4)
        colA.metric("Version", summary["version"])
        colB.metric("Nodes", summary["nodes"])
        colC.metric("Edges", summary["edges"])
        colD.metric("Measures", measure_count)

    # Small overview graph
    with st.expander("Overview graph", expanded=True):
        max_nodes = st.slider("Max nodes to render", 50, 400, 200, 25, key="max_nodes_overview")
        dot = get_graph_dot(max_nodes=max_nodes)
        try:
            st.graphviz_chart(dot, use_container_width=True)
        except Exception:
            st.info("Graphviz rendering not available.")

    # ------------- SECOND LEVEL: DRILL-DOWN -------------
    st.subheader("Drill-down by Component")

    # Node browser
    col1, col2 = st.columns([1,2])
    with col1:
        node_type = st.selectbox(
            "Filter by type",
            ["", "operator_state", "env_condition", "task", "operator_action", "plant_action", "unknown"],
            index=0,
            key="node_type_filter"
        )
        search = st.text_input("Search node id (contains)", key="node_search")
        nodes = list_nodes(node_type=node_type or None, search=search or None)
        node_ids = [n["node_id"] for n in nodes]
        selected_node = st.selectbox("Select a node", [""] + node_ids, index=0, key="selected_node")

        st.caption(f"{len(nodes)} match(es)")

    with col2:
        if selected_node:
            with st.container(border=True):
                st.markdown(f"### Node: `{selected_node}`")
                details = get_node_details(selected_node)
                if "error" in details:
                    st.error("Node not found")
                else:
                    st.markdown("**Attributes**")
                    pretty_json(details.get("data", {}))

                    radius = st.slider("Neighborhood radius", 1, 3, 1, 1, key="ego_radius")
                    dot_local = get_ego_graph_dot(selected_node, radius=radius, max_nodes=80)
                    st.graphviz_chart(dot_local, use_container_width=True)

                    col_in, col_out = st.columns(2)
                    with col_in:
                        st.markdown("**Incoming edges**")
                        st.dataframe(pd.DataFrame(details.get("in_edges", [])))
                    with col_out:
                        st.markdown("**Outgoing edges**")
                        st.dataframe(pd.DataFrame(details.get("out_edges", [])))

    # Edge browser with expandable rows
    st.subheader("Edges")
    edge_type = st.selectbox("Filter by edge type", ["", "causal", "mitigative", "impacts", "unknown"], index=0, key="edge_type_filter")
    edges = list_edges(edge_type=edge_type or None)
    st.caption(f"{len(edges)} edges")
    st.dataframe(pd.DataFrame(edges))

    # ---------- Pretty helpers (defined once) ----------
    if "_render_measure_card" not in globals():
        def _pill(text: str):
            st.markdown(
                "<span style='display:inline-block;padding:4px 10px;border-radius:999px;background:#eef2ff;border:1px solid #e5e7eb;font-size:12px;margin-right:6px'>"
                + str(text) + "</span>",
                unsafe_allow_html=True,
            )

        def _subpill(text: str):
            st.markdown(
                "<span style='display:inline-block;padding:2px 8px;border-radius:999px;background:#f1f5f9;border:1px solid #e5e7eb;font-size:11px;margin-right:4px'>"
                + str(text) + "</span>",
                unsafe_allow_html=True,
            )

        def _small(txt: str):
            st.markdown(f"<div style='color:#6b7280;font-size:12px'>{txt}</div>", unsafe_allow_html=True)

        def _render_thresholds(th_profiles):
            if not th_profiles:
                _small("No threshold profiles defined.")
                return
            for prof in th_profiles:
                st.markdown(f"**Profile:** `{prof.get('profile_id','unknown')}`")
                rngs = prof.get("ranges", [])
                if rngs:
                    df = pd.DataFrame(rngs)
                    st.dataframe(df, use_container_width=True, hide_index=True)
                st.divider()

        def _render_measure_card(m, mapping, thresholds, est_spec):
            # Header & summary
            st.markdown(f"### {m['measure_id']} — {m['name']}")
            _small(m.get("definition", ""))

            c1, c2, c3 = st.columns(3)
            c1.metric("Unit", m.get("unit", "—"))
            c2.metric("Aggregation", m.get("aggregation", "—"))
            c3.metric("Default confidence", m.get("confidence_default", "—"))

            # Linked PSFs
            st.markdown("**Linked PSFs:**")
            if m.get("linked_psfs"):
                cols = st.columns(min(4, len(m["linked_psfs"])))
                for i, psf in enumerate(m["linked_psfs"]):
                    with cols[i % len(cols)]:
                        _pill(psf)
            else:
                _small("None")

            st.divider()

            # Estimation
            st.markdown("#### Estimation")
            em = m.get("estimation_method", {}) or {}
            etype = em.get("type", "—")
            formula = em.get("formula")
            model_id = em.get("model_id")
            st.write(f"**Type:** `{etype}`")
            if formula:
                st.write(f"**Formula:** `{formula}`")
            if model_id:
                st.write(f"**Model:** `{model_id}`")

            # Inputs (deduped from estimation + mapping)
            inputs = set(em.get("inputs_required", []) or [])
            for it in mapping.get("inputs_map", []) or []:
                if isinstance(it, dict) and "signal" in it:
                    inputs.add(it["signal"])
            if inputs:
                st.write("**Inputs used:**")
                for s in sorted(inputs):
                    _subpill(s)
            else:
                _small("No inputs listed.")

            # Applicability & missing data policy
            st.divider()
            st.markdown("#### Applicability & Data Policy")
            app = mapping.get("applicability", {}) if isinstance(mapping, dict) else {}
            mdp = mapping.get("missing_data_policy", "—")
            colA, colB = st.columns(2)
            with colA:
                st.write("**Applicability**")
                if app:
                    for k, v in app.items():
                        st.write(f"- **{k}**: {', '.join(v) if isinstance(v, list) else v}")
                else:
                    _small("Not specified.")
            with colB:
                st.write("**Missing data policy**")
                st.write(f"`{mdp}`")

            # Thresholds tables
            st.divider()
            st.markdown("#### Thresholds")
            _render_thresholds(thresholds.get("threshold_profiles", []))

    # ------------- MEASURES & THRESHOLDS -------------
    st.subheader("Measures & Thresholds")
    meas = get_measure_catalog()["measures"]
    if meas:
        meas_labels = [f"{m['measure_id']} — {m['name']}" for m in meas]
        sel_meas = st.selectbox("Choose a measure", meas_labels, index=0)
        if sel_meas:
            mid = sel_meas.split(" — ")[0]
            _m = next(x for x in meas if x["measure_id"] == mid)
            _mapping = get_measure_mapping(mid)
            _thresholds = get_threshold_profiles(mid)
            _est_spec = get_estimation_spec(mid)
            _render_measure_card(_m, _mapping, _thresholds, _est_spec)
    else:
        st.caption("No measures found.")

    # ------------- REASONING / EXPLANATIONS -------------
    st.subheader("Recent Explanations")
    exps = get_recent_explanations(limit=10)
    if exps:
        for e in exps:
            with st.expander(f"Explanation {e['explanation_id']} — {e.get('timestamp','')}"):
                st.markdown("**Reasoning path**")
                for step in e.get("path", []):
                    st.markdown(f"- {step}")
                st.markdown("**Evidence**")
                pretty_json(e.get("evidence", {}))
    else:
        st.caption("No explanations yet. Run `infer_state()` from HPMS to generate.")

    # ------------- LIVE REASONING DEMO -------------
    st.subheader("Live Reasoning Demo")
    if "last_row" in st.session_state and st.session_state.get("last_row"):
        row = st.session_state["last_row"]
        payload = {
            "version": "v1",
            "timestamp": (row.get("timestamp").isoformat() if hasattr(row.get("timestamp"), "isoformat")
                        else datetime.datetime.utcnow().isoformat()),
            "operator_id": "opA",
            "task": {"id": row.get("task", "unknown"), "duration_s": int(row.get("task_duration", 0)), "mode": row.get("reactor_status", "normal")},
            "signals": {
                "hr_bpm": row.get("heart_rate", 0),
                "skin_temp": row.get("skin_temp", 0),
                "posture": row.get("posture", "unknown"),
                "eye_tracking": row.get("eye_tracking", "unknown"),
                "voice": row.get("voice", "unknown"),
                "emotion": row.get("face_emotion", "neutral"),
                "stress": row.get("face_stress", "low"),
                "room_temp": row.get("room_temp", 0),
                "cct_temp_k": row.get("cct_temp", 0),
                "light_intensity": row.get("light_intensity", 0),
                "humidity": row.get("humidity", 0),
                "pressure": row.get("pressure", 0),
            },
            "reactor_status": row.get("reactor_status", "normal"),
        }
        colR1, colR2 = st.columns(2)
        with colR1:
            st.caption("infer_state()")
            inf = infer_state(payload)
            pretty_json(inf)
        with colR2:
            st.caption("predict_state()")
            horizon = st.slider("Forecast horizon (seconds)", 60, 3600, 600, 60, key="forecast_h")
            pred = predict_state(payload, horizon_s=horizon)
            pretty_json(pred)
    else:
        st.info("Load a CSV in the HPMS Dashboard first to enable live reasoning demo.")
