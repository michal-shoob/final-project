from pyboolnet import file_exchange
import networkx as nx
from itertools import product

# Define the network in BNET format
bnet = """
v1,    1
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

initial_values = {
    "v1": None,
    "v2": 1,
    "v3": 1,
    "v4": None,
    "v5": 1,
    "v6": 1,
    "v7": None

}

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

    #Evaluates the next state of the network according to the primes and edge functions.
    
    new_state = state.copy()

    # Evaluate primes (node functions)
    print("\nEvaluating node functions:")
    for node, func in primes.items():
        if isinstance(func, str):  # if the function is a string
            try:
                # Do not replace AND and OR for node functions, as they are used for boolean logic
                new_state[node] = eval(func, {},new_state)
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
                new_state[end_node] = eval(edge_func, {}, new_state)
                print(f"Edge ({start_node} -> {end_node}): {edge_func} => {new_state[end_node]}")
            except Exception as e:
                print(f"Warning: Error evaluating edge function for edge ({start_node}, {end_node}). Error: {e}")
                new_state[end_node] = 0  # Default value

    return new_state



def find_initial_conditions(primes, target_values, nodes, edge_functions, initial_state, max_iterations=100, agent=None):
    """
    Find ONLY the initial conditions (including assignments for the None nodes)
    that lead the network to the target values after updates.

    :param primes: A dictionary of node functions (primes).
    :param target_values: A dictionary of target values for each node.
    :param nodes: A list of nodes in the network.
    :param edge_functions: A dictionary of edge functions.
    :param initial_state: A dictionary with initial values for each node (None => to be randomized).
    :param max_iterations: The maximum number of iterations to simulate (default: 100).
    :param agent: An optional Q-learning agent to guide the search (default: None).
    :return: A list of dictionaries, each containing:
        {
          "initial_condition": dict,        # The assignment (including your fixed values + bits for None)
          "final_state": dict,             # The final stable state after updates
          "intermediate_states": list[dict] # The states after each iteration until stabilization
        }
    """

    # Nodes with a value of None – we will generate all combinations for these nodes
    none_nodes = [node for node in nodes if initial_state.get(node) is None]

    # All possible combinations of 0/1 for the nodes that are None
    possible_replace_none = list(product([0, 1], repeat=len(none_nodes)))

    valid_initial_conditions = []
    visited_states = set()
    print("initial_state:", initial_state)
    print("none_nodes:", none_nodes)
    print("possible_replace_none:", possible_replace_none)

    for condition in possible_replace_none:
        # Build a new initial state for the current combination
        possible_conditions = initial_state.copy()
        for idx, node in enumerate(none_nodes):
            possible_conditions[node] = condition[idx]

        # Ensure we haven't already visited this state
        trial_state_key = tuple(sorted(possible_conditions.items()))
        if trial_state_key in visited_states:
            continue
        visited_states.add(trial_state_key)

        print(f"\n--- Testing initial condition: {possible_conditions} ---")
        current_state = possible_conditions.copy()
        intermediate_states = []

        # Update the network until it stabilizes or the maximum iterations are reached
        for _ in range(max_iterations):
            if agent:
                action = agent.choose_action(current_state)
                next_state = evaluate_state(current_state, primes, edge_functions)
            else:
                next_state = evaluate_state(current_state, primes, edge_functions)

            intermediate_states.append(next_state.copy())

            # If there is no change between the current state and the next state, the network has stabilized
            if next_state == current_state:
                print(f"--> Stabilized final state: {next_state}")
                # Remove the last state from the list since it's identical to the previous one
                intermediate_states.pop()
                break

            current_state = next_state

        # Check if the final state matches the target values
        if all(current_state[node] == val for node, val in target_values.items()):
            # If yes – add the valid initial condition to the list
            valid_initial_conditions.append({
                "initial_condition": possible_conditions.copy(),
                "final_state": current_state.copy(),
                "intermediate_states": intermediate_states
            })

    return valid_initial_conditions


