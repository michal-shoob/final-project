from boolean_network import primes_2, edge_functions_2, target_values_2, nodes_2,evaluate_state, find_initial_conditions,initial_values_2
from boolean_network import primes_3, edge_functions_3, target_values_3,initial_values_3, initial_values_3_1, nodes_3
from td_agent import TDAgent
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



def main():
    """
    Main function to run the simulation.
    """
   
    args = parse_arguments()
    logging.info(f"starting simulation...")
    # Create and train the TD learning agent
    td_agent = TDAgent(primes_3, edge_functions_3, target_values_3, initial_values_3_1, epsilon=args.epsilon)
    td_agent.train(episodes=1000)

   


if __name__ == "__main__":
    main()
