import networkx as nx
import os
from PIL import Image as PILImage  # For standalone scripts
from IPython.display import display, Image as IPythonImage
from boolean_network import primes, edge_functions, target_values, nodes,evaluate_state, find_initial_conditions,initial_values
from agent import Agent, next_graph_state
from q_learning_agent import QLearningAgent
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import logging
from logger_setup import logger



def parse_arguments():
    """
    Parse command-line arguments.

    :return: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Boolean Network Simulation")
    parser.add_argument("--epsilon", type=float, default=0.1, help="Epsilon value for exploration")
    return parser.parse_args()


def initialize_graph():
    """
    Initialize the graph with nodes and initial states.

    :return: The initialized graph.
    """
    # Create a directed graph
    graph = nx.DiGraph()

    # Add nodes with initial states
    for node in primes.keys():
        graph.add_node(node, state=0)  # Initialize all nodes to state 0

    # Add edges based on edge_functions
    for (start_node, end_node) in edge_functions.keys():
        graph.add_edge(start_node, end_node)

    return graph


def main():
    """
    Main function to run the simulation.
    """
    # Initialize the graph
    graph = initialize_graph()

    # Create an agent
    #agent = Agent(agent_id=1, graph=graph, position="V1")

    # Simulate the network for a few steps
    # num_steps = 10
    # for step in range(num_steps):
    #     print(f"\nStep {step + 1}:")
    #     print("Current node states:", nx.get_node_attributes(graph, "state"))
    #
    #     # Compute the next state of the graph
    #     graph = next_graph_state(graph, primes, edge_functions)
    #
    # print("\nFinal node states:", nx.get_node_attributes(graph, "state"))
    args = parse_arguments()

    # Create and train the Q-learning agent
    q_agent = QLearningAgent(primes, edge_functions, target_values, initial_values, epsilon=args.epsilon)
    q_agent.train(episodes=1000)

    # Find initial conditions using the trained Q-learning agent
    valid_initial_conditions = find_initial_conditions(primes, target_values, nodes, edge_functions,initial_values, max_iterations=100)

    # Print results
    # Function to display images
    def display_image(image_path):
        """
        Display an image in a Jupyter Notebook or open it in the default viewer for standalone scripts.
        """
        try:
            # Check if running in a Jupyter Notebook
            display(IPythonImage(filename=image_path))
        except ImportError:
            # Fallback for standalone scripts: open the image using the default viewer
            try:
                img = PILImage.open(image_path)
                img.show()
            except Exception as e:
                print(f"Could not display the image: {e}")

    if valid_initial_conditions:
        print(f"\n{len(valid_initial_conditions)} valid initial conditions found:")
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

        # Save the table as an image
        table_image_path = "table_visualization.png"
        plt.savefig(table_image_path, bbox_inches="tight", dpi=300)
        print(f"Table saved as '{table_image_path}'")

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
        fig, axes = plt.subplots(1, num_plots, figsize=(20, 8))  # Increased figure size

        # Function to color nodes based on state comparison
        def get_node_color(state, target_state=None):
            colors = []
            for node in nodes:
                if target_state and state[node] == target_state.get(node, None):  # Final state, color green
                    colors.append("lightgreen")
                elif state[node] != initial_condition.get(node, None):  # Changed from initial state, color orange
                    colors.append("darkorange")
                else:  # No change, color default
                    colors.append("lightskyblue")
            return colors

        # Plot the initial state graph
        axes[0].set_title("Initial State")
        nx.draw_networkx_nodes(graph, pos, node_size=1000, node_color="lightskyblue", ax=axes[0])
        nx.draw_networkx_labels(graph, pos, labels={node: f"{node} ({initial_condition[node]})" for node in nodes},
                                font_size=12, font_weight="bold", ax=axes[0])
        nx.draw_networkx_edges(graph, pos, arrowstyle="->", arrowsize=15, edge_color="gray",
                               connectionstyle="arc3,rad=0.1",
                               min_source_margin=15, min_target_margin=15, ax=axes[0])
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_functions, font_size=8, label_pos=0.6,
                                     bbox=dict(alpha=0),
                                     ax=axes[0])

        # Plot intermediate state graphs
        for i, state in enumerate(intermediate_states, 1):
            axes[i].set_title(f"Step {i}")
            state_colors = get_node_color(state)  # Get colors based on the current and initial state
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
        nx.draw_networkx_edges(graph, pos, arrowstyle="->", arrowsize=15, edge_color="gray",
                               connectionstyle="arc3,rad=0.1",
                               min_source_margin=15, min_target_margin=15, ax=axes[-1])

        # Hide axes and display the plots
        for ax in axes:
            ax.axis("off")

        # Adjust subplot spacing
        plt.subplots_adjust(left=0.05, right=0.95, top=0.85, bottom=0.15, wspace=0.3)

        # Save the plot to a file
        graph_image_path = "network_states.png"
        plt.savefig(graph_image_path, bbox_inches="tight", dpi=300)
        print(f"Graph saved as '{graph_image_path}'")

        # Function to display images

        display_image(table_image_path)

        # Display the saved graph image
        display_image(graph_image_path)
    else:
        print("No initial conditions found that lead to the target values.")


if __name__ == "__main__":
    main()
