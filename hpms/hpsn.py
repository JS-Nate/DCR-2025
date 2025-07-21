import networkx as nx
import json
from datetime import datetime
from analysis import SAFETY_MARGINS

# Initialize HPSN as a directed graph
hpsn = nx.DiGraph()

# Initialize HPSN with nodes and edges based on HPMS inputs
def initialize_hpsn():
    # Nodes: operator states, environmental conditions, tasks, actions
    nodes = [
        ("high_heart_rate", {"type": "operator_state"}),
        ("theta-dominant", {"type": "operator_state"}),
        ("alpha-suppressed", {"type": "operator_state"}),
        ("beta-dominant", {"type": "operator_state"}),
        ("alpha-dominant", {"type": "operator_state"}),
        ("angry", {"type": "operator_state"}),
        ("sad", {"type": "operator_state"}),
        ("fear", {"type": "operator_state"}),
        ("surprised", {"type": "operator_state"}),
        ("neutral", {"type": "operator_state"}),
        ("happy", {"type": "operator_state"}),
        ("high_temp", {"type": "env_condition"}),
        ("low_temp", {"type": "env_condition"}),
        ("high_humidity", {"type": "env_condition"}),
        ("low_humidity", {"type": "env_condition"}),
        ("high_light_temp", {"type": "env_condition"}),
        ("low_light_temp", {"type": "env_condition"}),
        ("high_light_intensity", {"type": "env_condition"}),
        ("low_light_intensity", {"type": "env_condition"}),
        ("high_pressure", {"type": "env_condition"}),
        ("low_pressure", {"type": "env_condition"}),
        ("emergency_shutdown", {"type": "task"}),
        ("take_deep_breaths", {"type": "operator_action"}),
        ("lower_room_temp", {"type": "plant_action"}),
        ("increase_room_temp", {"type": "plant_action"}),
        ("adjust_lighting", {"type": "plant_action"}),
        ("reduce_reactor_output", {"type": "plant_action"})
    ]
    hpsn.add_nodes_from(nodes)
    
    # Edges: causal, mitigative, and task impact relationships
    edges = [
        ("high_temp", "high_heart_rate", {"type": "causal", "weight": 0.6, "timestamp": str(datetime.now())}),
        ("high_humidity", "high_heart_rate", {"type": "causal", "weight": 0.5, "timestamp": str(datetime.now())}),
        ("angry", "high_heart_rate", {"type": "causal", "weight": 0.7, "timestamp": str(datetime.now())}),
        ("sad", "theta-dominant", {"type": "causal", "weight": 0.65, "timestamp": str(datetime.now())}),
        ("fear", "alpha-suppressed", {"type": "causal", "weight": 0.6, "timestamp": str(datetime.now())}),
        ("take_deep_breaths", "high_heart_rate", {"type": "mitigative", "weight": 0.5, "timestamp": str(datetime.now())}),
        ("lower_room_temp", "high_temp", {"type": "mitigative", "weight": 0.6, "timestamp": str(datetime.now())}),
        ("increase_room_temp", "low_temp", {"type": "mitigative", "weight": 0.6, "timestamp": str(datetime.now())}),
        ("adjust_lighting", "high_light_intensity", {"type": "mitigative", "weight": 0.5, "timestamp": str(datetime.now())}),
        ("adjust_lighting", "low_light_intensity", {"type": "mitigative", "weight": 0.5, "timestamp": str(datetime.now())}),
        ("high_heart_rate", "emergency_shutdown", {"type": "impacts", "weight": 0.8, "timestamp": str(datetime.now())}),
        ("theta-dominant", "emergency_shutdown", {"type": "impacts", "weight": 0.7, "timestamp": str(datetime.now())}),
        ("alpha-suppressed", "emergency_shutdown", {"type": "impacts", "weight": 0.75, "timestamp": str(datetime.now())})
    ]
    hpsn.add_edges_from(edges)

# Map CSV inputs to HPSN node identifiers using SAFETY_MARGINS
def map_inputs_to_nodes(row):
    nodes = []
    
    # Operator states
    if row["heart_rate"] > SAFETY_MARGINS["1.1_heart_rate"]["max"]:
        nodes.append("high_heart_rate")
    if row["eeg_signal"].lower() in SAFETY_MARGINS["1.2_eeg_signal"]["warning"]:
        nodes.append("theta-dominant")
    if row["eeg_signal"].lower() in SAFETY_MARGINS["1.2_eeg_signal"]["danger"]:
        nodes.append("alpha-suppressed")
    if row["eeg_signal"].lower() in ["alpha-dominant", "beta-dominant"]:
        nodes.append(row["eeg_signal"].lower())
    if row["face_emotion"].lower() in SAFETY_MARGINS["1.3_face_emotion"]["danger"]:
        nodes.append(row["face_emotion"].lower())
    if row["face_emotion"].lower() in SAFETY_MARGINS["1.3_face_emotion"]["safe"] + SAFETY_MARGINS["1.3_face_emotion"]["warning"]:
        nodes.append(row["face_emotion"].lower())
    
    # Environmental conditions
    if row["room_temp"] > SAFETY_MARGINS["2.1_room_temp"]["max"]:
        nodes.append("high_temp")
    elif row["room_temp"] < SAFETY_MARGINS["2.1_room_temp"]["min"]:
        nodes.append("low_temp")
    if row["humidity"] > 60:
        nodes.append("high_humidity")
    elif row["humidity"] < 30:
        nodes.append("low_humidity")
    if row["light_temp"] > 5500:
        nodes.append("high_light_temp")
    elif row["light_temp"] < 4500:
        nodes.append("low_light_temp")
    if row["light_intensity"] > 1000:
        nodes.append("high_light_intensity")
    elif row["light_intensity"] < 300:
        nodes.append("low_light_intensity")
    if row["pressure"] > 1020:
        nodes.append("high_pressure")
    elif row["pressure"] < 980:
        nodes.append("low_pressure")
    
    # Task
    nodes.append(row["task"].lower().replace(" ", "_"))
    
    return nodes

# Query HPSN for suggested actions
def query_hpsn(inputs):
    operator_states = [n for n in inputs if hpsn.nodes[n]["type"] == "operator_state"]
    env_conditions = [n for n in inputs if hpsn.nodes[n]["type"] == "env_condition"]
    tasks = [n for n in inputs if hpsn.nodes[n]["type"] == "task"]
    
    actions = []
    
    # Find mitigative actions for operator states
    for state in operator_states:
        if state in hpsn.nodes:
            for neighbor in hpsn.successors(state):
                edge_data = hpsn.get_edge_data(state, neighbor)
                if edge_data.get("type") == "mitigative" and hpsn.nodes[neighbor]["type"] == "operator_action":
                    actions.append((neighbor, edge_data["weight"]))
    
    # Find mitigative actions for environmental conditions
    for condition in env_conditions:
        if condition in hpsn.nodes:
            for neighbor in hpsn.successors(condition):
                edge_data = hpsn.get_edge_data(condition, neighbor)
                if edge_data.get("type") == "mitigative" and hpsn.nodes[neighbor]["type"] == "plant_action":
                    actions.append((neighbor, edge_data["weight"]))
    
    # Boost actions for states impacting tasks
    for task in tasks:
        if task in hpsn.nodes:
            for state in operator_states:
                if hpsn.has_edge(state, task) and hpsn.get_edge_data(state, task)["type"] == "impacts":
                    for action, weight in actions[:]:
                        if hpsn.has_edge(action, state):
                            actions.append((action, weight * 1.2))
    
    return sorted(set(actions), key=lambda x: x[1], reverse=True)

# Update HPSN based on Llama recommendations and feedback
def update_hpsn(recommendation_data, feedback=None):
    operator_actions = recommendation_data.get("operator_actions", [])
    plant_actions = recommendation_data.get("plant_actions", [])
    confidence = recommendation_data.get("confidence", 0.5)
    related_concepts = recommendation_data.get("related_concepts", [])
    
    # Add action nodes
    for action in operator_actions + plant_actions:
        action_id = action.lower().replace(" ", "_")
        if action_id not in hpsn.nodes:
            action_type = "operator_action" if action in operator_actions else "plant_action"
            hpsn.add_node(action_id, type=action_type, description=action)
    
    # Add or update edges
    for action in operator_actions:
        action_id = action.lower().replace(" ", "_")
        for concept in related_concepts:
            if concept in hpsn.nodes and hpsn.nodes[concept]["type"] == "operator_state":
                if hpsn.has_edge(action_id, concept):
                    current_weight = hpsn.get_edge_data(action_id, concept)["weight"]
                    new_weight = min(current_weight + (confidence * 0.1), 1.0)
                else:
                    new_weight = confidence * 0.5
                hpsn.add_edge(
                    action_id, concept,
                    type="mitigative", weight=new_weight, timestamp=str(datetime.now())
                )
    
    for action in plant_actions:
        action_id = action.lower().replace(" ", "_")
        for concept in related_concepts:
            if concept in hpsn.nodes and hpsn.nodes[concept]["type"] == "env_condition":
                if hpsn.has_edge(action_id, concept):
                    current_weight = hpsn.get_edge_data(action_id, concept)["weight"]
                    new_weight = min(current_weight + (confidence * 0.1), 1.0)
                else:
                    new_weight = confidence * 0.5
                hpsn.add_edge(
                    action_id, concept,
                    type="mitigative", weight=new_weight, timestamp=str(datetime.now())
                )
    
    # Adjust weights based on feedback
    if feedback and "effectiveness" in feedback:
        feedback_score = feedback["effectiveness"]
        for action in operator_actions + plant_actions:
            action_id = action.lower().replace(" ", "_")
            for concept in related_concepts:
                if hpsn.has_edge(action_id, concept):
                    current_weight = hpsn.get_edge_data(action_id, concept)["weight"]
                    new_weight = max(0.0, min(current_weight + (feedback_score - 0.5) * 0.2, 1.0))
                    hpsn.edges[action_id, concept]["weight"] = new_weight

# Save HPSN to file
def save_hpsn(filename="hpsn.json"):
    data = nx.node_link_data(hpsn)
    with open(filename, "w") as f:
        json.dump(data, f)

# Load HPSN from file
def load_hpsn(filename="hpsn.json"):
    global hpsn
    try:
        with open(filename, "r") as f:
            data = json.load(f)
        hpsn = nx.node_link_graph(data)
    except FileNotFoundError:
        initialize_hpsn()