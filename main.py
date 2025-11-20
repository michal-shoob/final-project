from boolean_network import primes_20, edge_functions_20, target_values_20, nodes_20, initial_values_20
from boolean_network import primes_50, edge_functions_50, target_values_50, initial_values_50, valid_action_50
from boolean_network import primes_100, edge_functions_100, target_values_100, initial_values_100
from td_agent import TDAgent
import argparse
import logging
from logger_setup import logger
from helper import compute_possible_targets



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
    
    target_values_50_possible = compute_possible_targets(valid_action_50, primes_50, edge_functions_50)
    logging.info(f"starting simulation for initial state: {initial_values_50} /n and target values: {target_values_50_possible}")
    # Create and train the TD learning agent
    td_agent = TDAgent(primes_50, edge_functions_50, target_values_50_possible, initial_values_50, epsilon=args.epsilon)
    td_agent.train(episodes=1000)
   #td_agent = TDAgent(primes_100, edge_functions_100, target_values_100, initial_values_100, epsilon=args.epsilon)
    #td_agent.train(episodes=1000)
    



   


if __name__ == "__main__":
    main()
