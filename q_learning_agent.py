import numpy as np
from itertools import product
import logging



class QLearningAgent:
    def __init__(self, primes, edge_functions, target_values, initial_values, alpha=0.1, gamma=0.9, epsilon=0.1):
        """
        Initialize the Q-learning agent.

        :param primes: A dictionary of node functions (primes).
        :param edge_functions: A dictionary of edge functions.
        :param target_values: A dictionary of target values for each node.
        :param alpha: Learning rate (default: 0.1).
        :param gamma: Discount factor (default: 0.9).
        :param epsilon: Exploration rate (default: 0.1).
        """
        self.primes = primes
        self.edge_functions = edge_functions
        self.target_values = target_values
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.initial_values = dict(initial_values)  # Convert to dictionary if it isn't
        self.current_state = initial_values.copy()  # Initialize current state

        # Identify controllable nodes (those with None in initial_values)
        self.controllable_nodes = [node for node, val in initial_values.items() if val is None]
        self.fixed_nodes = {node: val for node, val in initial_values.items() if val is not None}
        possible_replace_none = product([0,1], repeat=len(self.controllable_nodes))
        possible_actions = []
        visited_states = set()
        for condition in possible_replace_none:
            # Build a new initial state for the current combination
            possible_conditions = self.current_state.copy()
            for idx, node in enumerate(self.controllable_nodes):
                possible_conditions[node] = condition[idx]

            # Ensure we haven't already visited this state
            trial_state_key = tuple(sorted(possible_conditions.items()))
            if trial_state_key in visited_states:
                continue
            visited_states.add(trial_state_key)
            possible_actions.append(possible_conditions)
        self.action_space = possible_actions  # Store possible actions

        # Initialize Q-table with all possible state-action pairs
        self.q_table = self.initialize_q_table()

        #logging.debug(f"Initial Q-table: {self.q_table}")  # Debug

    def initialize_q_table(self):
        """Initialize Q-table considering only None-valued nodes for actions"""
        # Generate all possible states (2^n possible states)
        state_space = list(product([0, 1], repeat=len(self.primes)))
        q_table= {
            tuple(zip(self.primes, state)): 0.0 for state in state_space
        }
        return q_table

    def get_state_key(self, state):
        """
        Convert the state to a hashable key (e.g., tuple) for the Q-table.

        :param state: The current state of the Boolean network.
        :return: A hashable key representing the state.
        """
        return tuple((k, state[k]) for k in self.primes)  # Convert state dictionary to immutable tuple

    def get_action_key(self, action):
        """
        Convert the action to a hashable key (e.g., tuple) for the Q-table.

        :param action: The action taken.
        :return: A hashable key representing the action.
        """
        return tuple(sorted((k, v) for k, v in action.items()))

    def choose_action(self):
        """
        Choose an action using an epsilon-greedy policy.

        :param state: The current state of the Boolean network.
        :return: The chosen action.
        """
        # Get possible actions
        action_space = self.action_space
        # Debugging print
        #logging.debug(f"\n Possible actions: {len(action_space)}")
        # Evaluate actions based on their Q-values
        action_values = []
        for action in action_space:
            try:
                action_key = self.get_state_key(action)
                action_values.append(self.q_table[action_key])  # Default small value if not present)
            except Exception as e:
                logging.error(f"Error processing action {action}: {str(e)}")
                action_values.append(0.0)  # Fallback value

        # Epsilon-greedy action selection
        if np.random.rand() < self.epsilon:
            # Explore: choose a random action
            return np.random.choice(action_space)
        else:
            max_q = max(action_values)
            best_actions = [a for a, q in zip(action_space, action_values) if q == max_q]
            return np.random.choice(best_actions)

    def get_possible_actions(self, state=None):
        """
        Define possible actions based on the current state.

        :param state: The current state of the Boolean network.
        :return: A list of possible actions.
        """
        if state is None:
            state = self.current_state

        # Nodes with a value of None – we will generate all combinations for these nodes
        none_nodes = [node for node in state if state.get(node) is None]

        # All possible combinations of 0/1 for the nodes that are None
        possible_replace_none = list(product([0, 1], repeat=len(none_nodes)))
        possible_actions = []
        visited_states = set()
        for condition in possible_replace_none:
            # Build a new initial state for the current combination
            possible_conditions = state.copy()
            for idx, node in enumerate(none_nodes):
                possible_conditions[node] = condition[idx]

            # Ensure we haven't already visited this state
            trial_state_key = tuple(sorted(possible_conditions.items()))
            if trial_state_key in visited_states:
                continue
            visited_states.add(trial_state_key)
            possible_actions.append(possible_conditions)
        return possible_actions

    def evaluate_state(self, state):

        #Evaluates the next state of the network according to the primes and edge functions.
        
        new_state = state.copy()

        # Evaluate primes (node functions)
        #print("\nEvaluating node functions:")
        for node, func in self.primes.items():
            if isinstance(func, str):  # if the function is a string
                try:
                    # Do not replace AND and OR for node functions, as they are used for boolean logic
                    new_state[node] = eval(func, {},new_state)
                    #print(f"Node {node}: {func} => {new_state[node]}")
                except Exception as e:
                    print(f"Warning: Error evaluating function for node {node}. Error: {e}")
                    new_state[node] = 0  # Default value
            elif callable(func):  # if the function is callable
                new_state[node] = func(state)
                #print(f"Node {node} callable: {new_state[node]}")
            else:  # Here we handle the case where the function is a simple value (like a constant)
                new_state[node] = state[node]  # The state doesn't change for simple nodes
                #print(f"Node {node} simple value: {new_state[node]}")

        # Now, evaluate the edge functions (between nodes)
        #print("\nEvaluating edge functions:")
        for (start_node, end_node), edge_func in self.edge_functions.items():
            # Check if start_node is in the current state and the end_node is also in the state
            if start_node in state:
                try:
                    # Keep the AND and OR as they are (no need to replace)
                    new_state[end_node] = eval(edge_func, {}, new_state)
                    #print(f"Edge ({start_node} -> {end_node}): {edge_func} => {new_state[end_node]}")
                except Exception as e:
                    print(f"Warning: Error evaluating edge function for edge ({start_node}, {end_node}). Error: {e}")
                    new_state[end_node] = 0  # Default value

        return new_state
    

    def get_reward(self, state, next_state, step_penalty=0):
        """
        Calculate the reward based on the target values and distance to goal.

        :param state: The current state.
        :param next_state: The next state.
        :param step_penalty: Penalty for number of steps taken.
        :return: The reward.
        """
        if self.is_terminal_state(next_state):
            # Success: base reward minus step penalty (fewer steps = higher reward)
            base_reward = 10.0
            reward = base_reward - (step_penalty * 0.1)  # Small penalty per step
        else:
            # Calculate distance to target (how many nodes match target)
            matching_nodes = sum(1 for node, target in self.target_values.items() 
                               if next_state.get(node) == target)
            total_nodes = len(self.target_values)
            distance_reward = (matching_nodes / total_nodes) * 2.0  # Partial progress reward
            reward = distance_reward - 0.5  # Small negative base to encourage progress
        return reward

    def update_q_table(self, state, reward, next_state):
        """
        Update the Q-table using the Q-learning update rule.

        :param state: The current state.
        :param reward: The reward received.
        :param next_state: The next state.
        """
        # Convert states and actions to hashable keys
        state_key = self.get_state_key(state)  # Use the passed state, not current_state
        # action_key = self.get_action_key(action)
        next_state_key = self.get_state_key(next_state)

        # Debugging print
        #logging.info("\n--- Updating Q-table ---")  # Debug
        #print(f"State: {state}, Action: {action}, Reward: {reward}, Next State: {next_state}")

        # Get current Q-value (initialize if not present) - Q(s):
        current_q = self.q_table.get((state_key), 0.0)  # Default small value

        next_state_value = self.q_table.get((next_state_key), 0.0)  # Default small value

        # Q-learning update
        # new V(s) <- V(s) + alpha [R + gamma * V(s')-V(s)]
        new_q = current_q + self.alpha * (reward + self.gamma * next_state_value - current_q)
        self.q_table[(state_key)] = new_q
        logging.debug(f"\n Q-table in state {state_key} updated to: {new_q}")  # Debug


    def train(self, episodes):
        """
        Train the Q-learning agent over a number of episodes.

        :param episodes: The number of episodes to train.
        """
        for episode in range(episodes):
            done = False
            # Choose an action
            action = self.choose_action()
            if action == self.current_state:
                done =  True
                logging.info(f"\n--- action {action} led to the target state immediately ---")
                break    
            self.current_state = action     # Use the action to set the current state
            logging.debug(f"\n action checks: {action}")  # Debug
            steps = 0
            intermediate_states = [action]
            while not done and steps < 500:  # Limit steps to avoid infinite loops
                # Evaluate the next state using the Boolean network update rules
                next_state = self.evaluate_state(self.current_state)
                 # Store the intermediate state
                intermediate_states.append(next_state.copy())
                
                next_state_key = self.get_state_key(next_state)
                if self.q_table.get(next_state_key) >= 0.0: # If the next state can lead to target
                    reward = self.get_reward(self.current_state, next_state, steps)

                    # Update the Q-table
                    self.update_q_table(self.current_state, reward, next_state)

                    # Check if the state has stabilized or reached target
                    done = (self.current_state == next_state) or self.is_terminal_state(next_state)

                    # Transition to the next state
                    self.current_state = next_state
                    steps += 1
                else:
                    done = True  # Stop if the next state cannot lead to target

            # After the episode ends, apply final rewards based on success/failure
            if self.is_terminal_state(self.current_state):
                # SUCCESS: Give positive rewards with step-based penalty
                # Shorter paths get higher rewards
                base_success_reward = 10.0
                step_penalty = len(intermediate_states) * 0.1
                final_reward = base_success_reward - step_penalty
                
                # Update intermediate states with success rewards (higher for states closer to goal)
                for i in range(len(intermediate_states)):
                    current_state = intermediate_states[i]
                    if i < len(intermediate_states) - 1:
                        next_state = intermediate_states[i + 1]
                    else:
                        next_state = self.current_state
                    
                    # Reward decreases with distance from goal (later states get higher rewards)
                    distance_from_goal = len(intermediate_states) - i - 1
                    state_reward = final_reward - (distance_from_goal * 0.5)
                    self.update_q_table(current_state, state_reward, next_state)
            else:
                # FAILURE: Give negative rewards to discourage these paths
                failure_penalty = -5.0
                
                # Update intermediate states with negative rewards
                for i in range(len(intermediate_states)):
                    current_state = intermediate_states[i]
                    if i < len(intermediate_states) - 1:
                        next_state = intermediate_states[i + 1]
                    else:
                        next_state = self.current_state
                    
                    # Apply failure penalty to discourage these intermediate states
                    self.update_q_table(current_state, failure_penalty, next_state)
            self.current_state= action.copy()  # Store the last action taken
        if not done:
            print("not found")


    def get_initial_state(self):
        """
        Define the initial state of the network.

        :return: The initial state.
        """
        return {node: v for node, v in self.initial_values.items()}

    def is_terminal_state(self, state=None):
        """
        Check if the state is terminal (i.e., matches the target values).

        :return: True if the state is terminal, False otherwise.
        """
        if state is None:
            state = self.current_state

        terminal = all(state.get(node, None) == target for node, target in self.target_values.items())
        #logging.debug(f"\nTerminal state check: {terminal}")  # Debug
        return terminal

    def get_optimal_policy(self):
        """
        Extract the optimal policy from the Q-table.

        :return: A dictionary mapping states to optimal actions.
        """
        policy = {}
        for (state_key, action), q_value in self.q_table.items():
            if state_key not in policy or q_value > self.q_table.get((state_key, policy[state_key]), 0):
                policy[state_key] = action
        return policy
