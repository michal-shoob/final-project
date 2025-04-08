import numpy as np
from boolean_network import primes, edge_functions, target_values, nodes, evaluate_state, find_initial_conditions
from itertools import product


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

        # Initialize Q-table with all possible state-action pairs
        self.q_table = self.initialize_q_table()
        print(f"Initial Q-table: {self.q_table}")  # Debug

    def initialize_q_table(self):
        """Initialize Q-table considering only None-valued nodes for actions"""
        q_table = {}

        # Generate all possible network states (2^n possible states)
        all_nodes = list(self.primes.keys())
        state_space = list(product([0, 1], repeat=len(all_nodes)))
        initial_state_tuple = tuple(self.initial_values)
        # Ensure the initial state is included in the state space
        if initial_state_tuple not in state_space:
            state_space.append(initial_state_tuple)
        # Remove duplicates
        state_space = list(set(state_space))

        # Generate action space only for controllable nodes
        action_space = list(product([0, 1], repeat=len(self.controllable_nodes)))

        for state_values in state_space:
            state = dict(zip(all_nodes, state_values))
            state_key = self.get_state_key(state)  # Use current state, not self.current_state

            for action_values in action_space:
                # Create action dict only for controllable nodes
                action = dict(zip(self.controllable_nodes, action_values))
                action_key = self.get_action_key(action)  # Only need controllable nodes here

                # Initialize Q-values
                #if self.is_terminal_state(state):  # Pass the state we're evaluating
                #    q_table[(state_key, action_key)] = 0.0  # Terminal state
                #else:
                    # Initialize with small positive values to encourage exploration
                q_table[(state_key, action_key)] = 0.0

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

    def choose_action(self, state=None):
        """
        Choose an action using an epsilon-greedy policy.

        :param state: The current state of the Boolean network.
        :return: The chosen action.
        """
        if state is None:
            state = self.current_state

        # Get possible actions based on the current state
        possible_actions = self.get_possible_actions(state)
        state_key = self.get_state_key(state)
        # Debugging print
        print(f"\nState: {state_key}, Possible actions: {len(possible_actions)}")

        if np.random.rand() < self.epsilon:
            # Explore: choose a random action
            return np.random.choice(possible_actions)
        else:
            # Evaluate actions based on their Q-values
            action_values = []
            for action in possible_actions:
                action_key = self.get_action_key(action)
                action_values.append(self.q_table.get((state_key, action_key), 0.1))

            max_q = max(action_values)
            best_actions = [a for a, q in zip(possible_actions, action_values) if q == max_q]
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

    def get_reward(self, state, action, next_state):
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
        print(f"\nReward calculated: {reward}")  # Debug
        return reward

    def update_q_table(self, state, action, reward, next_state):
        """
        Update the Q-table using the Q-learning update rule.

        :param state: The current state.
        :param action: The action taken.
        :param reward: The reward received.
        :param next_state: The next state.
        """
        # Convert states and actions to hashable keys
        state_key = self.get_state_key(state)  # Use the passed state, not current_state
        action_key = self.get_action_key(action)
        next_state_key = self.get_state_key(next_state)

        # Debugging print
        #print("\n--- Updating Q-table ---")  # Debug
        #print(f"State: {state}, Action: {action}, Reward: {reward}, Next State: {next_state}")

        # Get current Q-value (initialize if not present) - Q(s,a):
        current_q = self.q_table.get((state_key, action_key), 0.0)  # Default small value

        # Calculate max Q-value for next state- max_a Q(s',a):
        next_actions = self.get_possible_actions(next_state)
        if next_actions:
            max_next_q = max(
                [self.q_table.get((next_state_key, self.get_action_key(a)), 0.0)
                 for a in next_actions]
            )
        else:
            max_next_q = 0.0
        # Q-learning update
        # new Q(s,a) <- Q(s,a) + alpha [R + gamma * max_a Q(s',a)-Q(s,a)]
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.q_table[(state_key, action_key)] = new_q

        # Debug output
        print(f"\n--- Q-table Update ---")
        print(f"State: {state}")
        print(f"Action: {action}")
        print(f"Reward: {reward}")
        print(f"Next state: {next_state}")
        print(f"Current Q: {current_q:.3f}")
        print(f"Max next Q: {max_next_q:.3f}")
        print(f"Updated Q: {new_q:.3f}")

    def train(self, episodes):
        """
        Train the Q-learning agent over a number of episodes.

        :param episodes: The number of episodes to train.
        """
        for episode in range(episodes):
            self.current_state = self.get_initial_state()  # Get the initial state of the network
            done = False

            # Choose an action
            action = self.choose_action(self.current_state)
            i = 0
            while not done and i < 10:
                # Take the action and observe the next state and reward
                if i == 0:
                    # Apply action to get new state
                    next_state = action.copy()
                else:
                    # Evaluate the next state using the Boolean network update rules
                    next_state = evaluate_state(self.current_state, self.primes, self.edge_functions)
                reward = self.get_reward(self.current_state, action, next_state)

                # Update the Q-table
                self.update_q_table(self.current_state, action, reward, next_state)

                # Check if the state has stabilized
                done = (self.current_state == next_state)

                # Transition to the next state
                self.current_state = next_state.copy()
                i += 1

            # After the episode ends, check if the final state is terminal
            if self.is_terminal_state(self.current_state):
                final_reward = 1.0
            else:
                final_reward = -1.0

            # Update Q-table for the last action taken
            initial_state = self.get_initial_state()  # Get the initial state of the network
            self.update_q_table(initial_state, action, final_reward, action)

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
        print(f"\nTerminal state check: {terminal}")  # Debug
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
