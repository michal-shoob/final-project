import networkx as nx
from boolean_network import evaluate_state


class Agent:
    def __init__(self, agent_id, graph, position="V1"):
        """
        Initialize the agent.

        :param agent_id: A unique identifier for the agent.
        :param graph: The graph representing the Boolean network.
        :param position: The initial position of the agent (default: "V1").
        """
        self.agent_id = agent_id
        self.position = position
        self.graph = graph


def next_graph_state(current_graph, primes, edge_functions):
    """
    Compute the next state of the graph based on the current state and update rules.

    :param current_graph: The current graph with node states.
    :param primes: A dictionary of node functions (primes).
    :param edge_functions: A dictionary of edge functions.
    :return: The updated graph with the next state.
    """
    # Get current node states
    current_state = nx.get_node_attributes(current_graph, "state")

    # Compute next state based on update rules
    next_state = evaluate_state(current_state, primes, edge_functions)

    # Create a copy of the graph and update node states
    next_graph = current_graph.copy()
    nx.set_node_attributes(next_graph, next_state, "state")

    return next_graph
