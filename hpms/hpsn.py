
# HPSN Module (Module 5) – Consolidated with Explorer
from __future__ import annotations
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Optional

import networkx as nx

# Try to import analysis helpers if available
try:
    from analysis import SAFETY_MARGINS, compute_psf_loads, compute_output_score
except Exception:
    SAFETY_MARGINS = {
        "1.1_heart_rate": {"max": 1000},
        "1.2_skin_temp": {"max": 1000},
        "1.3_face_emotion": {"danger": [], "warning": [], "safe": []},
        "2.1_room_temp": {"min": -273, "max": 1000},
        "2.2_cct_temp": {"min": 0, "max": 100000},
        "2.3_light_intensity": {"min": 0, "max": 1e9},
        "2.4_humidity": {"min": 0, "max": 100},
        "2.5_pressure": {"min": 0, "max": 1e6},
    }
    def compute_psf_loads(*args, **kwargs):
        return {"physiological":0.2,"behavioral":0.2,"interaction":0.2,"environmental":0.2,"task_system":0.2}, 0.2
    def compute_output_score(*args, **kwargs):
        return 0.85

VERSION = "hpsn-v1.5"
hpsn = nx.DiGraph()
_EXPLANATIONS: Dict[str, Dict[str, Any]] = {}

_MEASURE_CATALOG: List[Dict[str, Any]] = [
    {
        "measure_id": "M001",
        "name": "task_accuracy",
        "definition": "Proportion of correctly executed task steps in window",
        "unit": "fraction",
        "aggregation": "rolling_mean_5min",
        "linked_psfs": ["attention", "workload"],
        "estimation_method": {"type": "equation", "formula": "1 - error_rate",
                              "inputs_required": ["error_events", "task_events"]},
        "confidence_default": 0.9,
        "provenance": {"ontology_version": VERSION, "source": "module5"}
    },
    {
        "measure_id": "M004",
        "name": "situational_awareness_index",
        "definition": "Composite index of attention to plant state and alarms",
        "unit": "score_0_1",
        "aggregation": "instant",
        "linked_psfs": ["workload", "fatigue"],
        "estimation_method": {"type": "model", "model_id": "BN_SA_2025_07",
                              "inputs_required": ["alarm_rate", "eye_fix_stability", "procedure_step_complexity"]},
        "confidence_default": 0.78,
        "provenance": {"ontology_version": VERSION, "source": "module5"}
    },
    {
        "measure_id": "M012",
        "name": "decision_latency_ms",
        "definition": "Median time from alarm onset to committed decision",
        "unit": "ms",
        "aggregation": "median_5min",
        "linked_psfs": ["workload", "stress", "attention"],
        "estimation_method": {"type": "equation", "formula": "t_decide - t_alarm",
                              "inputs_required": ["t_alarm", "t_decide"]},
        "confidence_default": 0.8,
        "provenance": {"ontology_version": VERSION, "source": "module5"}
    },
]

_INPUT_MAPPINGS: Dict[str, Dict[str, Any]] = {
    "M004": {
        "inputs_map": [
            {"signal": "alarm_rate_per_min", "transform": "minmax_0_12", "weight": 0.42},
            {"signal": "eye_fix_stability", "transform": "zscore_op_baseline", "weight": 0.33},
            {"signal": "procedure_step_complexity", "transform": "categorical_map:{low:0,med:0.5,high:1}", "weight": 0.25},
        ],
        "applicability": {"plant_mode": ["abnormal", "transient"], "tasks": ["alarm_triage", "diagnosis"]},
        "missing_data_policy": "degrade_confidence_and_renorm_weights",
    },
    "M012": {
        "inputs_map": [
            {"signal": "t_alarm", "transform": "identity", "weight": 0.5},
            {"signal": "t_decide", "transform": "identity", "weight": 0.5},
        ],
        "applicability": {"plant_mode": ["normal", "abnormal", "transient"], "tasks": ["diagnosis", "response"]},
        "missing_data_policy": "drop_sample_if_missing",
    },
}

_THRESHOLDS: Dict[str, List[Dict[str, Any]]] = {
    "M004": [
        {"profile_id": "default", "ranges": [
            {"label": "optimal", "min": 0.75, "max": 1.0},
            {"label": "watch",   "min": 0.55, "max": 0.75},
            {"label": "critical","min": 0.0,  "max": 0.55},
        ]},
        {"profile_id": "alarm_triage_highload", "ranges": [
            {"label": "optimal", "min": 0.80, "max": 1.0},
            {"label": "watch",   "min": 0.60, "max": 0.80},
            {"label": "critical","min": 0.0,  "max": 0.60},
        ]},
    ],
    "M012": [
        {"profile_id": "default", "ranges": [
            {"label": "optimal", "min": 0, "max": 1200},
            {"label": "watch", "min": 1200, "max": 2500},
            {"label": "critical", "min": 2500, "max": 60000},
        ]}
    ]
}

_ESTIMATION_SPECS: Dict[str, Dict[str, Any]] = {
    "M004": {
        "type": "bayesian_network",
        "model_id": "BN_SA_2025_07",
        "nodes": ["workload", "attention", "alarm_rate", "eye_fix_stability", "sa_index"],
        "explain_endpoint": "/v1/hpsn/explain?model_id=BN_SA_2025_07"
    },
    "M012": {
        "type": "equation",
        "model_id": "DLAT_v1",
        "formula": "decision_latency_ms = t_decide - t_alarm"
    }
}

def initialize_hpsn() -> None:
    nodes = [
        ("high_heart_rate", {"type": "operator_state"}),
        ("high_skin_temp", {"type": "operator_state"}),
        ("high_stress", {"type": "operator_state"}),
        ("medium_stress", {"type": "operator_state"}),
        ("sad", {"type": "operator_state"}),
        ("fear", {"type": "operator_state"}),
        ("angry", {"type": "operator_state"}),
        ("surprised", {"type": "operator_state"}),
        ("neutral", {"type": "operator_state"}),
        ("happy", {"type": "operator_state"}),
        ("high_temp", {"type": "env_condition"}),
        ("low_temp", {"type": "env_condition"}),
        ("high_humidity", {"type": "env_condition"}),
        ("low_humidity", {"type": "env_condition"}),
        ("high_cct", {"type": "env_condition"}),
        ("low_cct", {"type": "env_condition"}),
        ("high_light_intensity", {"type": "env_condition"}),
        ("low_light_intensity", {"type": "env_condition"}),
        ("high_pressure", {"type": "env_condition"}),
        ("low_pressure", {"type": "env_condition"}),
        ("emergency_shutdown", {"type": "task"}),
        ("monitoring", {"type": "task"}),
        ("adjusting_controls", {"type": "task"}),
        ("reporting", {"type": "task"}),
        ("take_deep_breaths", {"type": "operator_action"}),
        ("lower_room_temp", {"type": "plant_action"}),
        ("increase_room_temp", {"type": "plant_action"}),
        ("adjust_lighting", {"type": "plant_action"}),
        ("reduce_reactor_output", {"type": "plant_action"}),
    ]
    hpsn.add_nodes_from(nodes)

    now = str(datetime.now())
    edges = [
        ("high_temp", "high_heart_rate", {"type": "causal", "weight": 0.6, "timestamp": now}),
        ("high_humidity", "high_heart_rate", {"type": "causal", "weight": 0.5, "timestamp": now}),
        ("angry", "high_heart_rate", {"type": "causal", "weight": 0.7, "timestamp": now}),
        ("sad", "high_stress", {"type": "causal", "weight": 0.65, "timestamp": now}),
        ("fear", "high_stress", {"type": "causal", "weight": 0.6, "timestamp": now}),
        ("take_deep_breaths", "high_heart_rate", {"type": "mitigative", "weight": 0.5, "timestamp": now}),
        ("lower_room_temp", "high_temp", {"type": "mitigative", "weight": 0.6, "timestamp": now}),
        ("increase_room_temp", "low_temp", {"type": "mitigative", "weight": 0.6, "timestamp": now}),
        ("adjust_lighting", "high_light_intensity", {"type": "mitigative", "weight": 0.5, "timestamp": now}),
        ("adjust_lighting", "low_light_intensity", {"type": "mitigative", "weight": 0.5, "timestamp": now}),
        ("high_heart_rate", "emergency_shutdown", {"type": "impacts", "weight": 0.8, "timestamp": now}),
        ("high_stress", "emergency_shutdown", {"type": "impacts", "weight": 0.75, "timestamp": now}),
    ]
    hpsn.add_edges_from(edges)

def save_hpsn(filename: str = "hpsn.json") -> None:
    data = nx.node_link_data(hpsn)
    with open(filename, "w") as f:
        json.dump(data, f)

def load_hpsn(filename: str = "hpsn.json") -> None:
    global hpsn
    try:
        with open(filename, "r") as f:
            data = json.load(f)
        hpsn = nx.node_link_graph(data)
    except FileNotFoundError:
        initialize_hpsn()

def map_inputs_to_nodes(row: Dict[str, Any]) -> List[str]:
    nodes: List[str] = []
    try:
        if float(row.get("heart_rate", 0)) > SAFETY_MARGINS["1.1_heart_rate"]["max"]:
            nodes.append("high_heart_rate")
        if float(row.get("skin_temp", 0)) > SAFETY_MARGINS["1.2_skin_temp"]["max"]:
            nodes.append("high_skin_temp")
        stress = str(row.get("face_stress", "")).lower()
        if stress == "high":
            nodes.append("high_stress")
        elif stress == "medium":
            nodes.append("medium_stress")
        emotion = str(row.get("face_emotion", "neutral")).lower()
        nodes.append(emotion)
        temp = float(row.get("room_temp", 0))
        if temp > SAFETY_MARGINS["2.1_room_temp"]["max"]:
            nodes.append("high_temp")
        elif temp < SAFETY_MARGINS["2.1_room_temp"]["min"]:
            nodes.append("low_temp")
        if float(row.get("humidity", 0)) > SAFETY_MARGINS["2.4_humidity"]["max"]:
            nodes.append("high_humidity")
        elif float(row.get("humidity", 0)) < SAFETY_MARGINS["2.4_humidity"]["min"]:
            nodes.append("low_humidity")
        cct = float(row.get("cct_temp", 0))
        if cct > SAFETY_MARGINS["2.2_cct_temp"]["max"]:
            nodes.append("high_cct")
        elif cct < SAFETY_MARGINS["2.2_cct_temp"]["min"]:
            nodes.append("low_cct")
        light = float(row.get("light_intensity", 0))
        if light > SAFETY_MARGINS["2.3_light_intensity"]["max"]:
            nodes.append("high_light_intensity")
        elif light < SAFETY_MARGINS["2.3_light_intensity"]["min"]:
            nodes.append("low_light_intensity")
        pres = float(row.get("pressure", 0))
        if pres > SAFETY_MARGINS["2.5_pressure"]["max"]:
            nodes.append("high_pressure")
        elif pres < SAFETY_MARGINS["2.5_pressure"]["min"]:
            nodes.append("low_pressure")
        nodes.append(str(row.get("task", "unknown")).lower().replace(" ", "_"))
    except Exception:
        pass
    return nodes

def query_hpsn(inputs: List[str]) -> List[Tuple[str, float]]:
    operator_states = [n for n in inputs if n in hpsn.nodes and hpsn.nodes[n]["type"] == "operator_state"]
    env_conditions  = [n for n in inputs if n in hpsn.nodes and hpsn.nodes[n]["type"] == "env_condition"]
    tasks           = [n for n in inputs if n in hpsn.nodes and hpsn.nodes[n]["type"] == "task"]

    actions: List[Tuple[str, float]] = []
    for state in operator_states:
        for neighbor in hpsn.predecessors(state):
            edge_data = hpsn.get_edge_data(neighbor, state)
            if edge_data and edge_data.get("type") == "mitigative" and hpsn.nodes[neighbor]["type"] == "operator_action":
                actions.append((neighbor, float(edge_data.get("weight", 0.5))))
    for condition in env_conditions:
        for neighbor in hpsn.predecessors(condition):
            edge_data = hpsn.get_edge_data(neighbor, condition)
            if edge_data and edge_data.get("type") == "mitigative" and hpsn.nodes[neighbor]["type"] == "plant_action":
                actions.append((neighbor, float(edge_data.get("weight", 0.5))))
    for task in tasks:
        for state in operator_states:
            if hpsn.has_edge(state, task) and hpsn.get_edge_data(state, task).get("type") == "impacts":
                boosted: List[Tuple[str, float]] = []
                for action, weight in actions:
                    if hpsn.has_edge(action, state):
                        boosted.append((action, float(weight) * 1.2))
                    else:
                        boosted.append((action, float(weight)))
                actions = boosted
    uniq: Dict[str, float] = {}
    for a, w in actions:
        uniq[a] = max(uniq.get(a, 0.0), float(w))
    return sorted(uniq.items(), key=lambda x: x[1], reverse=True)

def update_hpsn(recommendation_data: Dict[str, Any], feedback: Optional[Dict[str, Any]] = None) -> None:
    operator_actions = recommendation_data.get("operator_actions", [])
    plant_actions = recommendation_data.get("plant_actions", [])
    confidence = float(recommendation_data.get("confidence", 0.5))
    related_concepts = recommendation_data.get("related_concepts", [])
    for action in operator_actions + plant_actions:
        action_id = action.lower().replace(" ", "_")
        if action_id not in hpsn.nodes:
            action_type = "operator_action" if action in operator_actions else "plant_action"
            hpsn.add_node(action_id, type=action_type)
    for action in operator_actions:
        action_id = action.lower().replace(" ", "_")
        for concept in related_concepts:
            if concept in hpsn.nodes and hpsn.nodes[concept]["type"] == "operator_state":
                current_weight = hpsn.get_edge_data(action_id, concept, default={}).get("weight", 0.0)
                new_weight = min(current_weight + (confidence * 0.1 if current_weight else confidence * 0.5), 1.0)
                hpsn.add_edge(action_id, concept, type="mitigative", weight=new_weight, timestamp=str(datetime.now()))
    for action in plant_actions:
        action_id = action.lower().replace(" ", "_")
        for concept in related_concepts:
            if concept in hpsn.nodes and hpsn.nodes[concept]["type"] == "env_condition":
                current_weight = hpsn.get_edge_data(action_id, concept, default={}).get("weight", 0.0)
                new_weight = min(current_weight + (confidence * 0.1 if current_weight else confidence * 0.5), 1.0)
                hpsn.add_edge(action_id, concept, type="mitigative", weight=new_weight, timestamp=str(datetime.now()))
    if feedback and "effectiveness" in feedback:
        feedback_score = float(feedback["effectiveness"])
        for action in operator_actions + plant_actions:
            action_id = action.lower().replace(" ", "_")
            for concept in related_concepts:
                if hpsn.has_edge(action_id, concept):
                    current_weight = float(hpsn.get_edge_data(action_id, concept).get("weight", 0.0))
                    new_weight = max(0.0, min(current_weight + (feedback_score - 0.5) * 0.2, 1.0))
                    hpsn.edges[action_id, concept]["weight"] = new_weight

# ------ Explorer & Introspection ------

def get_structure_report() -> Dict[str, Any]:
    node_count = hpsn.number_of_nodes()
    edge_count = hpsn.number_of_edges()
    type_counts: Dict[str, int] = {}
    for n, attrs in hpsn.nodes(data=True):
        t = attrs.get("type", "unknown")
        type_counts[t] = type_counts.get(t, 0) + 1
    edge_type_counts: Dict[str, int] = {}
    for u, v, attrs in hpsn.edges(data=True):
        et = attrs.get("type", "unknown")
        edge_type_counts[et] = edge_type_counts.get(et, 0) + 1
    orphans = [n for n in hpsn.nodes() if hpsn.degree(n) == 0]
    hubs = sorted([(n, hpsn.degree(n)) for n in hpsn.nodes()], key=lambda x: x[1], reverse=True)[:10]
    return {
        "version": VERSION,
        "nodes": node_count,
        "edges": edge_count,
        "node_types": type_counts,
        "edge_types": edge_type_counts,
        "orphans": orphans,
        "top_hubs": hubs,
        "measure_count": len(_MEASURE_CATALOG),
        "measures": [m["name"] for m in _MEASURE_CATALOG],
        "thresholded_measures": list(_THRESHOLDS.keys()),
    }

def list_nodes(node_type: Optional[str] = None, search: Optional[str] = None) -> List[Dict[str, Any]]:
    results = []
    patt = search.lower() if search else None
    for n, attrs in hpsn.nodes(data=True):
        if node_type and attrs.get("type") != node_type:
            continue
        if patt and patt not in str(n).lower():
            continue
        results.append({"id": n, "type": attrs.get("type", "unknown")})
    return results

def list_edges(edge_type: Optional[str] = None) -> List[Dict[str, Any]]:
    res = []
    for u, v, attrs in hpsn.edges(data=True):
        if edge_type and attrs.get("type") != edge_type:
            continue
        res.append({"src": u, "dst": v, "type": attrs.get("type", "unknown"), "weight": attrs.get("weight"), "timestamp": attrs.get("timestamp")})
    return res

def get_node_details(node_id: str) -> Dict[str, Any]:
    if node_id not in hpsn.nodes:
        return {"error": "unknown_node"}
    preds = [{"id": p, "edge": hpsn.get_edge_data(p, node_id)} for p in hpsn.predecessors(node_id)]
    succs = [{"id": s, "edge": hpsn.get_edge_data(node_id, s)} for s in hpsn.successors(node_id)]
    return {"id": node_id, "type": hpsn.nodes[node_id].get("type", "unknown"), "in_edges": preds, "out_edges": succs}

def get_recent_explanations(limit: int = 20) -> List[Dict[str, Any]]:
    items = list(_EXPLANATIONS.items())
    def ts_of(v):
        try:
            return v[1].get("timestamp", "")
        except Exception:
            return ""
    items.sort(key=ts_of, reverse=True)
    out = []
    for k, v in items[:limit]:
        out.append({"explanation_id": k, **v})
    return out

def get_version_info() -> Dict[str, Any]:
    return {"version": VERSION, "catalog_size": len(_MEASURE_CATALOG)}

# ------ Config artifacts (HPMS Function 1) ------

def get_measure_catalog() -> Dict[str, Any]:
    return {"version": VERSION, "catalog_generated_at": datetime.utcnow().isoformat() + "Z", "measures": _MEASURE_CATALOG}

def get_measure_mapping(measure_id: str) -> Dict[str, Any]:
    return {"measure_id": measure_id, **_INPUT_MAPPINGS.get(measure_id, {"inputs_map": [], "applicability": {}, "missing_data_policy": "ignore"})}

def get_threshold_profiles(measure_id: str) -> Dict[str, Any]:
    return {"measure_id": measure_id, "threshold_profiles": _THRESHOLDS.get(measure_id, [])}

def get_estimation_spec(measure_id: str) -> Dict[str, Any]:
    return {"measure_id": measure_id, "estimation_spec": _ESTIMATION_SPECS.get(measure_id, {"type": "unknown"})}

def get_ontology_delta(since: Optional[datetime] = None) -> Dict[str, Any]:
    return {"diff_id": f"ontodelta_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}", "adds": [], "updates": [], "removes": []}

def get_config_bundle() -> Dict[str, Any]:
    bundle = get_measure_catalog()
    for m in bundle["measures"]:
        mid = m["measure_id"]
        m["inputs_map"] = _INPUT_MAPPINGS.get(mid, {}).get("inputs_map", [])
        m["threshold_profiles"] = _THRESHOLDS.get(mid, [])
        m["estimation_spec"] = _ESTIMATION_SPECS.get(mid, {})
    bundle["diff"] = get_ontology_delta()
    return bundle

# ------ Reasoning/runtime ------

def _label_from_score(x: float, cuts: Tuple[float, float] = (0.33, 0.66)) -> str:
    if x >= cuts[1]: return "high"
    if x >= cuts[0]: return "elevated"
    return "low"

def infer_state(payload: Dict[str, Any]) -> Dict[str, Any]:
    signals = payload.get("signals", {})
    hr = float(signals.get("hr_bpm", signals.get("heart_rate", 0)))
    skin_temp = float(signals.get("skin_temp", 0))
    posture = str(signals.get("posture", "unknown"))
    eye_tracking = str(signals.get("eye_tracking", "unknown"))
    voice = str(signals.get("voice", "unknown"))
    emotion = str(signals.get("emotion", "neutral")).lower()
    stress = str(signals.get("stress", "low")).lower()
    room_temp = float(signals.get("room_temp", 0))
    cct_temp = float(signals.get("cct_temp_k", signals.get("cct_temp", 0)))
    light_intensity = float(signals.get("light_intensity", 0))
    humidity = float(signals.get("humidity", 0))
    pressure = float(signals.get("pressure", 0))
    task = str(payload.get("task", {}).get("id", payload.get("task", "unknown")))
    task_duration = int(payload.get("task", {}).get("duration_s", payload.get("task_duration", 0)))
    reactor_status = str(payload.get("task", {}).get("mode", payload.get("reactor_status", "normal")))

    per_group, input_load = compute_psf_loads(
        hr, skin_temp, posture, eye_tracking, voice, emotion, stress,
        room_temp, cct_temp, light_intensity, humidity, pressure, task, task_duration, reactor_status
    )

    workload_score = per_group.get("task_system", 0.0)
    stress_proxy = max(per_group.get("physiological", 0.0), 0.0)
    attention_proxy = max(0.0, 1.0 - (per_group.get("behavioral", 0.0) + workload_score) / 2.0)

    state = {
        "workload": _label_from_score(workload_score),
        "stress": _label_from_score(stress_proxy),
        "attention": ("low" if attention_proxy < 0.4 else "medium" if attention_proxy < 0.7 else "high"),
        "error_risk": round((workload_score * 0.5 + stress_proxy * 0.3 + (1.0 - attention_proxy) * 0.2), 3)
    }

    output_score = compute_output_score(fallback_task_duration=task_duration or 120.0)
    measures = {
        "task_accuracy_p50": round(max(0.0, min(1.0, output_score)), 3),
        "resp_time_ms": int(800 + 1200 * (workload_score + stress_proxy) / 2.0),
    }

    explanation_id = f"exp_{uuid.uuid4().hex[:8]}"
    _EXPLANATIONS[explanation_id] = {
        "path": [
            f"task_system load={workload_score:.2f} + physiological load={stress_proxy:.2f} → stress/workload↑",
            f"behavioral load + workload → attention change (proxy={attention_proxy:.2f})",
            f"composite → error_risk={state['error_risk']}"
        ],
        "evidence": {
            "per_group_loads": per_group,
            "input_load": input_load,
            "signals_used": ["hr_bpm","skin_temp","posture","eye_tracking","voice","emotion","stress",
                             "room_temp","cct_temp_k","light_intensity","humidity","pressure","task","task_duration"]
        },
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }

    return {"version": VERSION, "state": state, "measures": measures, "per_group_loads": per_group,
            "input_load": round(input_load, 3), "explanation_id": explanation_id}

def predict_state(payload: Dict[str, Any], horizon_s: int = 600) -> Dict[str, Any]:
    base = infer_state(payload)
    per_group = dict(base.get("per_group_loads", {}))
    creep = min(0.15, horizon_s / 3600.0)
    per_group["task_system"] = min(1.0, per_group.get("task_system", 0.0) + creep)
    projected = {
        "workload": _label_from_score(per_group["task_system"]),
        "stress": base["state"]["stress"],
        "attention": base["state"]["attention"],
        "error_risk": round(min(1.0, base["state"]["error_risk"] + creep * 0.5), 3)
    }
    return {"version": VERSION, "horizon_s": horizon_s, "state": projected, "per_group_loads": per_group}

def recommend_actions(context: Dict[str, Any]) -> Dict[str, Any]:
    # kept for backward compatibility (HPMS now owns action selection)
    row_like = {
        "heart_rate": context.get("signals", {}).get("hr_bpm", 0),
        "skin_temp": context.get("signals", {}).get("skin_temp", 0),
        "face_stress": context.get("signals", {}).get("stress", "low"),
        "face_emotion": context.get("signals", {}).get("emotion", "neutral"),
        "room_temp": context.get("signals", {}).get("room_temp", 0),
        "cct_temp": context.get("signals", {}).get("cct_temp_k", 0),
        "light_intensity": context.get("signals", {}).get("light_intensity", 0),
        "humidity": context.get("signals", {}).get("humidity", 0),
        "pressure": context.get("signals", {}).get("pressure", 0),
        "task": context.get("task", {}).get("id", "unknown"),
    }
    nodes = map_inputs_to_nodes(row_like)
    ranked = query_hpsn(nodes)
    enriched = []
    for action, weight in ranked[:8]:
        enriched.append({
            "action_id": action,
            "type": hpsn.nodes[action]["type"] if action in hpsn.nodes else "unknown",
            "expected_benefit": round(min(1.0, float(weight)), 2),
            "confidence": round(0.6 + 0.4 * min(1.0, float(weight)), 2),
            "targets": [t for t in hpsn.successors(action)]
        })
    return {"version": VERSION, "recommendations": enriched, "related_nodes": nodes}

def explain(explanation_id: str) -> Dict[str, Any]:
    return _EXPLANATIONS.get(explanation_id, {"error": "unknown_explanation_id"})





# ========= HPSN Explorer / Introspection (safe to add) =========

def get_version_info() -> dict:
    return {
        "version": VERSION,
        "nodes": hpsn.number_of_nodes(),
        "edges": hpsn.number_of_edges(),
        "node_types": sorted({hpsn.nodes[n].get("type", "unknown") for n in hpsn.nodes}),
    }

def get_structure_report(top_k: int = 10) -> dict:
    import itertools
    # Degree hubs
    deg = [(n, hpsn.degree(n)) for n in hpsn.nodes]
    top_hubs = sorted(deg, key=lambda x: x[1], reverse=True)[:top_k]
    # Orphans
    orphans = [n for n in hpsn.nodes if hpsn.degree(n) == 0]
    # Edge types count
    edge_type_counts = {}
    for u, v, d in hpsn.edges(data=True):
        et = d.get("type", "unknown")
        edge_type_counts[et] = edge_type_counts.get(et, 0) + 1
    # Node type counts
    node_type_counts = {}
    for n, d in hpsn.nodes(data=True):
        nt = d.get("type", "unknown")
        node_type_counts[nt] = node_type_counts.get(nt, 0) + 1
    return {
        "summary": get_version_info(),
        "node_type_counts": node_type_counts,
        "edge_type_counts": edge_type_counts,
        "top_hubs": top_hubs,
        "orphans": orphans,
    }

def list_nodes(node_type: str | None = None, search: str | None = None) -> list[dict]:
    results = []
    search_low = (search or "").lower()
    for n, d in hpsn.nodes(data=True):
        if node_type and d.get("type") != node_type:
            continue
        if search and search_low not in str(n).lower():
            continue
        results.append({"node_id": n, "type": d.get("type", "unknown")})
    return sorted(results, key=lambda x: (x["type"], x["node_id"]))

def list_edges(edge_type: str | None = None) -> list[dict]:
    out = []
    for u, v, d in hpsn.edges(data=True):
        if edge_type and d.get("type") != edge_type:
            continue
        out.append({
            "src": u, "dst": v,
            "type": d.get("type", "unknown"),
            "weight": d.get("weight", None),
            "timestamp": d.get("timestamp", None),
        })
    return out

def get_node_details(node_id: str) -> dict:
    if node_id not in hpsn.nodes:
        return {"error": "node_not_found"}
    data = hpsn.nodes[node_id]
    succ = []
    for v in hpsn.successors(node_id):
        d = hpsn.get_edge_data(node_id, v)
        succ.append({"to": v, **d})
    pred = []
    for u in hpsn.predecessors(node_id):
        d = hpsn.get_edge_data(u, node_id)
        pred.append({"from": u, **d})
    return {"node_id": node_id, "data": data, "out_edges": succ, "in_edges": pred}

def get_recent_explanations(limit: int = 20) -> list[dict]:
    # _EXPLANATIONS holds {id: {..., 'timestamp': ISO}}
    items = []
    for k, v in _EXPLANATIONS.items():
        items.append({"explanation_id": k, **v})
    items.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    return items[:limit]

def get_graph_dot(max_nodes: int = 250) -> str:
    """
    Build a simple DOT graph (compatible with st.graphviz_chart).
    We keep it small to avoid huge render times.
    """
    nodes = list(hpsn.nodes)[:max_nodes]
    node_set = set(nodes)
    lines = ["digraph HPSN {", 'rankdir=LR;', 'node [shape=box, fontsize=10];']
    # Node styling by type
    for n in nodes:
        t = hpsn.nodes[n].get("type", "unknown")
        label = f"{n}\\n({t})"
        lines.append(f'"{n}" [label="{label}"];')
    # Edges between included nodes
    for u, v, d in hpsn.edges(data=True):
        if u in node_set and v in node_set:
            et = d.get("type", "edge")
            w = d.get("weight", "")
            edge_label = f"{et}{f' {w:.2f}' if isinstance(w, (int,float)) else ''}"
            lines.append(f'"{u}" -> "{v}" [label="{edge_label}", fontsize=9];')
    lines.append("}")
    return "\n".join(lines)
