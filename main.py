from graph1 import evaluate_state
import networkx as nx
class Agent:
    def __init__(self, agent_id, graph, position="V1" ):
        self.agent_id = agent_id
        self.position = position
        self.graph = graph


def next_graph_state(current_graph, primes, edge_functions):


    # Get current node states
    current_state = nx.get_node_attributes(current_graph, "state")

    # Compute next state based on update rules
    next_state = evaluate_state(current_state, primes, edge_functions)

    # Create a copy of the graph and update node states
    next_graph = current_graph.copy()
    nx.set_node_attributes(next_graph, next_state, "state")

    return next_graph
