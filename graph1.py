from pyboolnet import file_exchange
import networkx as nx
import matplotlib.pyplot as plt
from itertools import product

# Define the network in BNET format
bnet = """
v1,    v1
v2,    v2
v3,    v1 | v4
v4,    v6 & v2
v5,    v1 & v2 & v3
v6,    v4 & v5
v7,    v5 | (v6 & v4)
"""

# Convert the network to primes format
primes = file_exchange.bnet2primes(bnet)

# List of nodes
nodes = list(primes.keys())

# Define target values
target_values = {
    "v1": 1,
    "v2": 1,
    "v3": 1,
    "v4": 1,
    "v5": 1,
    "v6": 1,
    "v7": 1
}

# Define edge functions
edge_functions = {
    ("v1", "v3"): "v1 | v4",
    ("v4", "v3"): "v6 & v2",
    ("v1", "v5"): "v1 & v2 & v3",
    ("v2", "v5"): "v1 & v2 & v3",
    ("v3", "v5"): "v1 & v2 & v3",
    ("v4", "v6"): "v4 & v5",
    ("v5", "v6"): "v4 & v5",
    ("v5", "v7"): "v5 | (v6 & v4)",
    ("v6", "v7"): "v5 | (v6 & v4)",
    ("v2", "v6"): "v2 | v4",
    ("v3", "v7"): "v3 & v5"
}


# Create a graph with NetworkX
graph = nx.DiGraph()
graph.add_nodes_from(nodes)
graph.add_edges_from(edge_functions.keys())

# Function to evaluate the next state
def evaluate_state(state, primes, edge_functions):
    new_state = state.copy()
    for node, func in primes.items():
        if isinstance(func, str):
            try:
                new_state[node] = eval(func, {}, state)
            except Exception as e:
                new_state[node] = 0  # Default value in case of error
    for (start_node, end_node), edge_func in edge_functions.items():
        if start_node in state:
            try:
                new_state[end_node] = eval(edge_func, {}, state)
            except Exception as e:
                new_state[end_node] = 0  # Default value in case of error
    return new_state

# Function to draw all states
def draw_all_states_with_intermediates(valid_initial_conditions, graph, edge_functions, target_values, nodes):
    initial_state = valid_initial_conditions[0][0]
    final_state = target_values
    intermediate_states = []

    # Evaluate intermediate states
    current_state = initial_state.copy()
    while current_state != final_state:
        next_state = evaluate_state(current_state, primes, edge_functions)
        intermediate_states.append(next_state)
        if next_state == current_state:
            break  # Stabilized
        current_state = next_state

    # Define total number of graphs (initial + intermediates + final)
    total_states = [initial_state] + intermediate_states + [final_state]
    num_states = len(total_states)

    # Define positions and create subplots
    pos = nx.spring_layout(graph)  # Define consistent layout
    fig, axes = plt.subplots(1, num_states, figsize=(6 * num_states, 6))

    for idx, (state, ax) in enumerate(zip(total_states, axes)):
        # Define node colors for each state
        node_colors = ["lightgreen" if state[node] == 1 else "lightskyblue" for node in nodes]

        # Draw nodes, edges, and labels
        nx.draw_networkx_nodes(graph, pos, node_size=1000, node_color=node_colors, ax=ax)
        nx.draw_networkx_labels(graph, pos, labels={node: f"{node} ({state[node]})" for node in nodes}, font_size=12, ax=ax)
        nx.draw_networkx_edges(graph, pos, ax=ax, arrowstyle="->", arrowsize=15, edge_color="gray", connectionstyle="arc3,rad=0.1")
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_functions, font_size=8, label_pos=0.6, ax=ax)

        # Add titles for each graph
        if idx == 0:
            ax.set_title("Initial State")
        elif idx == num_states - 1:
            ax.set_title("Final State")
        else:
            ax.set_title(f"Intermediate State {idx}")

        ax.axis("off")  # Turn off axis

    plt.tight_layout()
    plt.show()

# Function to find initial conditions
def find_initial_conditions(primes, target_values, nodes, edge_functions, max_iterations=100):
    """
    Find initial conditions that lead the network to the target values after a number of updates.
    """
    possible_conditions = list(product([0, 1], repeat=len(nodes)))
    valid_initial_conditions = []

    for initial_conditions in possible_conditions:
        current_state = dict(zip(nodes, initial_conditions))

        # Perform updates to the network until it stabilizes or the max number of iterations is reached
        for _ in range(max_iterations):
            new_state = evaluate_state(current_state, primes, edge_functions)

            # If the state has stabilized, we break out of the loop
            if new_state == current_state:
                break

            current_state = new_state

        # If the final state matches the target values, we store the initial condition
        if all(current_state[node] == value for node, value in target_values.items()):
            valid_initial_conditions.append((dict(zip(nodes, initial_conditions)), current_state))

    return valid_initial_conditions

# Find initial conditions that lead to the target values
valid_initial_conditions = find_initial_conditions(primes, target_values, nodes, edge_functions)

# Draw the graphs including all intermediate states
if valid_initial_conditions:
    draw_all_states_with_intermediates(valid_initial_conditions, graph, edge_functions, target_values, nodes)
else:
    print("No valid initial conditions found that lead to the target values.")
