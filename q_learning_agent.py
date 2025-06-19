import numpy as np
from boolean_network import primes_2, edge_functions_2, target_values_2, nodes_2, evaluate_state, find_initial_conditions
from itertools import product
import logging
from logger_setup import logger



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
        possible_replace_none = list(product([0, 1], repeat=len(self.controllable_nodes)))
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
        return tuple(sorted(state.items()))  # Convert state dictionary to immutable tuple

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
        none_nodes = [node for node in nodes if state.get(node) is None]

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

    def get_reward(self, state, next_state):
        """
        Calculate the reward based on the target values.

        :param state: The current state.
        :param action: The action taken.
        :param next_state: The next state.
        :return: The reward.
        """
        reward = 0
        if self.is_terminal_state(next_state):
            reward = 1.0  # Terminal state reward
        else:
            reward = -1.0  # Non-terminal state reward
            # for node, target in self.target_values.items():
            #     if next_state[node] == target:
            #         reward = (1 / len(self.target_values))/10  # Normalized reward
        #logging.debug(f"\nReward calculated: {reward}")  # Debug
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
            if done:
                logging.info(f"\n--- action {action} led to the target state immediately ---")
                break
            self.current_state = action     # Use the action to set the current state
            logging.debug(f"\n action checks: {action}")  # Debug
            steps = 0
            intermediate_states = [action]
            while not done and steps < 20:
                # Evaluate the next state using the Boolean network update rules
                next_state = evaluate_state(self.current_state, self.primes, self.edge_functions)
                reward = self.get_reward(self.current_state, next_state)

                # Update the Q-table
                self.update_q_table(self.current_state, reward, next_state)

                # Store the intermediate state
                intermediate_states.append(next_state.copy())

                # Check if the state has stabilized
                done = (self.current_state == next_state)

                # Transition to the next state
                self.current_state = next_state.copy()
                steps += 1

            # After the episode ends, check if the final state is terminal
            if self.is_terminal_state(self.current_state):
                final_reward = 1.0
            # If not terminal, assign a negative reward
            else:
                final_reward = -1.0

            # for state in intermediate_states:
                # Update Q-table for each intermediate state
            #    self.update_q_table(state, final_reward, state)
            # Update Q-table for the last action taken
            for i in range(len(intermediate_states)):
                current_state = intermediate_states[i]
                if i < len(intermediate_states) - 1:
                    next_state = intermediate_states[i + 1]
                else:
                    next_state = self.current_state  # Use the final state
                self.update_q_table(current_state, final_reward, next_state)
            self.current_state= action.copy()  # Store the last action taken




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
