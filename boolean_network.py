from pyboolnet import file_exchange
import networkx as nx
import matplotlib.pyplot as plt
from itertools import product
import pandas as pd

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

# Define edge functions (to keep them for completeness)
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


def evaluate_state(state, primes, edge_functions):
    """
    Evaluates the next state of the network according to the primes and edge functions.
    """
    new_state = state.copy()

    # Evaluate primes (node functions)
    print("\nEvaluating node functions:")
    for node, func in primes.items():
        if isinstance(func, str):  # if the function is a string
            try:
                # Do not replace AND and OR for node functions, as they are used for boolean logic
                new_state[node] = eval(func, {}, state)
                print(f"Node {node}: {func} => {new_state[node]}")
            except Exception as e:
                print(f"Warning: Error evaluating function for node {node}. Error: {e}")
                new_state[node] = 0  # Default value
        elif callable(func):  # if the function is callable
            new_state[node] = func(state)
            print(f"Node {node} callable: {new_state[node]}")
        else:  # Here we handle the case where the function is a simple value (like a constant)
            new_state[node] = state[node]  # The state doesn't change for simple nodes
            print(f"Node {node} simple value: {new_state[node]}")

    # Now, evaluate the edge functions (between nodes)
    print("\nEvaluating edge functions:")
    for (start_node, end_node), edge_func in edge_functions.items():
        # Check if start_node is in the current state and the end_node is also in the state
        if start_node in state:
            try:
                # Keep the AND and OR as they are (no need to replace)
                new_state[end_node] = eval(edge_func, {}, state)
                print(f"Edge ({start_node} -> {end_node}): {edge_func} => {new_state[end_node]}")
            except Exception as e:
                print(f"Warning: Error evaluating edge function for edge ({start_node}, {end_node}). Error: {e}")
                new_state[end_node] = 0  # Default value

    return new_state


def find_initial_conditions(primes, target_values, nodes, edge_functions, max_iterations=100):
    """
    Find initial conditions that lead the network to the target values after a number of updates.
    """
    possible_conditions = list(product([0, 1], repeat=len(nodes)))
    valid_initial_conditions = []

    for initial_conditions in possible_conditions:
        print(f"\nTesting initial condition: {initial_conditions}")
        current_state = dict(zip(nodes, initial_conditions))
        intermediate_states = []

        # Perform updates to the network until it stabilizes or the max number of iterations is reached
        for _ in range(max_iterations):
            new_state = evaluate_state(current_state, primes, edge_functions)
            # Save the new state
            intermediate_states.append(new_state.copy())

            # If the state has stabilized, we break out of the loop
            if new_state == current_state:
                print(f"The final state of these initial conditions:{list(current_state.values())}")
                intermediate_states.pop()
                break

            current_state = new_state

        # If the final state matches the target values, we store the initial condition
        if all(current_state[node] == value for node, value in target_values.items()):
            valid_initial_conditions.append({
                "initial_condition": dict(zip(nodes, initial_conditions)),
                "final_state": current_state,
                "intermediate_states": intermediate_states
            })

    return valid_initial_conditions


# Finding initial conditions that lead to the target values
valid_initial_conditions = find_initial_conditions(primes, target_values, nodes, edge_functions)

# Printing results
if valid_initial_conditions:
    print(f"{len(valid_initial_conditions)} valid initial conditions found:")
    for i, condition in enumerate(valid_initial_conditions, 1):
        print(f"Initial condition {i}: {condition['initial_condition']} => Final state: {condition['final_state']}")

    # --print a table--
    # Create an empty list to store data for the table
    table_data = []
    step_labels = []

    # Iterate over valid initial conditions and their intermediate states
    for condition in valid_initial_conditions:
        initial_condition = condition["initial_condition"]
        intermediate_states = condition["intermediate_states"]

        # Add the initial state as the first row, with step number "Initial State"
        table_data.append(list(initial_condition.values()) + ["Initial State"])
        step_labels.append("Initial State")

        # Create a list of values for each intermediate state (node values only) and their step number
        for step, state in enumerate(intermediate_states, 1):  # Start from step 1
            table_data.append(list(state.values()) + [f"Step {step}"])
            step_labels.append(f"Step {step}")

        # Add the final state as the last row
        final_state = condition["final_state"]
        table_data.append(list(final_state.values()) + ["Final State"])
        step_labels.append("Final State")

    # Add a column for step labels and create a DataFrame from the intermediate states data
    df = pd.DataFrame(table_data, columns=nodes + ["Step"])
    df.insert(0, "Step Number", step_labels)  # Insert the step number as the first column

    # Remove the last column (Step) before plotting
    df = df.drop(columns=["Step"])

    # Set column widths for readability
    col_widths = [max(df[col].apply(lambda x: len(str(x)))) for col in df.columns]

    # Plot the table using Matplotlib with enhanced readability
    plt.figure(figsize=(18, 7))  # Adjust the size for better visibility
    plt.axis('tight')
    plt.axis('off')

    # Plot the table with enhanced styling
    table = plt.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center',
                      colColours=["lightyellow"] * len(df.columns))

    # Adjust font size and table properties dynamically based on available space
    num_rows, num_cols = df.shape
    # Set lower limits for readability
    min_font_size = 8  # Minimum font size for readability
    min_cell_width = 0.05  # Minimum width for cells
    min_cell_height = 0.04  # Minimum height for cells
    # Adjust font size and table properties dynamically based on available space
    cell_width = max(1.0 / num_cols, min_cell_width)  # Ensure cell width does not go below the limit
    cell_height = max(1.0 / num_rows, min_cell_height)  # Ensure cell height does not go below the limit

    # Set the font size dynamically based on the size of the table
    font_size = max(min_font_size, 12 - int(num_rows / 10))  # Ensure font size stays above the minimum
    for (i, j), cell in table.get_celld().items():
        # Adjust cell size
        cell.set_width(cell_width)
        cell.set_height(cell_height)

        # Adjust font size for better readability
        cell.set_fontsize(font_size)

    # Apply bold and color to the "Initial State" and "Final State"
    for i, label in enumerate(step_labels):
        # Apply bold and light blue for "Initial State"
        if label == "Initial State":
            for j in range(len(df.columns)):
                table[(i + 1, j)].set_text_props(weight='bold')
                table[(i + 1, j)].set_facecolor('lightblue')  # Light blue color for initial state
        # Apply bold and light green for "Final State"
        elif label == "Final State":
            for j in range(len(df.columns)):
                table[(i + 1, j)].set_text_props(weight='bold')
                table[(i + 1, j)].set_facecolor('lightgreen')  # Light green color for final state

    # Show the plot
    plt.tight_layout()
    plt.show()

    # --print a graph--
    # Select the first valid initial condition (assuming it's already sorted or chosen)
    first_condition = valid_initial_conditions[0]
    initial_condition = first_condition["initial_condition"]
    intermediate_states = first_condition["intermediate_states"]
    final_state = first_condition["final_state"]

    # Create a directed graph based on the nodes and edge functions
    edges = list(edge_functions.keys())
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)

    # Define node positions using spring layout
    pos = nx.spring_layout(graph)

    # Create subplots for each step (initial state + intermediate steps + final state)
    num_plots = len(intermediate_states) + 2  # Initial state + intermediate steps + final state
    fig, axes = plt.subplots(1, num_plots, figsize=(18, 8))

    # Function to color nodes based on state comparison
    def get_node_color(state, target_state=None):
        colors = []

        if state is None:
            # If state is None, return an empty color list (or handle as needed)
            return colors

        for node in nodes:
            initial_state = initial_condition.get(node, None)
            if target_state and state[node] == target_state.get(node, None):  # Final state, color green
                colors.append("lightgreen")
            elif state[node] != initial_state:  # Changed from last step, color red
                colors.append("darkorange")
            else:
                # No change, color default
                colors.append("lightskyblue")
        return colors


    # Plot the initial state graph
    axes[0].set_title("Initial State")
    nx.draw_networkx_nodes(graph, pos, node_size=1000, node_color="lightskyblue", ax=axes[0])
    nx.draw_networkx_labels(graph, pos, labels={node: f"{node} ({initial_condition[node]})" for node in nodes},
                            font_size=12, font_weight="bold", ax=axes[0])
    nx.draw_networkx_edges(graph, pos, arrowstyle="->", arrowsize=15, edge_color="gray", connectionstyle="arc3,rad=0.1",
                           min_source_margin=15, min_target_margin=15, ax=axes[0])
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_functions, font_size=8, label_pos=0.6, bbox=dict(alpha=0),
                                 ax=axes[0])

    # Plot intermediate state graphs
    previous_state = initial_condition  # Start from the initial state
    for i, state in enumerate(intermediate_states, 1):
        axes[i].set_title(f"Step {i}")
        state_colors = get_node_color(state)  # Get colors based on the current and previous state
        nx.draw_networkx_nodes(graph, pos, node_size=1000, node_color=state_colors, ax=axes[i])
        nx.draw_networkx_labels(graph, pos, labels={node: f"{node} ({state[node]})" for node in nodes},
                                font_size=12, font_weight="bold", ax=axes[i])
        nx.draw_networkx_edges(graph, pos, arrowstyle="->", arrowsize=15, edge_color="gray",
                               connectionstyle="arc3,rad=0.1",
                               min_source_margin=15, min_target_margin=15, ax=axes[i])

    # Plot the final state graph
    axes[-1].set_title("Final State (Target Values)")
    nx.draw_networkx_nodes(graph, pos, node_size=1000, node_color="lightgreen", ax=axes[-1])
    nx.draw_networkx_labels(graph, pos, labels={node: f"{node} ({final_state[node]})" for node in nodes},
                            font_size=12, font_weight="bold", ax=axes[-1])
    nx.draw_networkx_edges(graph, pos, arrowstyle="->", arrowsize=15, edge_color="gray", connectionstyle="arc3,rad=0.1",
                           min_source_margin=15, min_target_margin=15, ax=axes[-1])

    # Hide axes and display the plots
    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.show()
else:
    print("No initial conditions found that lead to the target values.")
