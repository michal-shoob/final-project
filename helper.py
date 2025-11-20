from boolean_network import evaluate_state

def compute_possible_targets(valid_actions, primes, edge_functions):
    """
    Compute the possible targets for a given initial state using the Boolean network update rules.
    """
    current_values= valid_actions
    done = False
    while not done:
            
        # Evaluate the next state using the Boolean network update rules
        next_state = evaluate_state(current_values, primes, edge_functions)
        # Check if the state has stabilized or reached target
        done = (current_values == next_state)

        # Transition to the next state
        current_values = next_state
              
    return current_values



