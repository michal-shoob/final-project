import numpy as np
from boolean_network import primes, edge_functions, target_values, nodes, evaluate_state, find_initial_conditions
from boolean_network import initial_values
from itertools import product

class QLearningAgent:
    def __init__(self, primes, edge_functions, target_values, alpha=0.1, gamma=0.9, epsilon=0.1):
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

        # Initialize Q-table as a dictionary
        self.q_table = {}  # Key: (state, action), Value: Q-value
        for k, v in initial_values.items():
            # if p already assigns a value, keep it (1 for this value, and 0 for the other)
            if v is not None:
                self.q_table[(k, 0)] = 1 if v == 0 else 0  #If the value is 0, it sets a 100% probability (Q-value = 1) for action 0 and 0% for action 1
                self.q_table[(k, 1)] = 1 if v == 1 else 0 #If the value is 1, it sets a 100% probability (Q-value = 1) for action 0 and 0% for action 1
            else:
                self.q_table[(k, 0)] = 0.5
                self.q_table[(k, 1)] = 0.5


    def get_state_key(self, state):
        """
        Convert the state to a hashable key (e.g., tuple) for the Q-table.

        :param state: The current state of the Boolean network.
        :return: A hashable key representing the state.
        """
        return tuple(state.items())  # Convert dictionary to tuple of items

    def choose_action(self, state):
        """
        Choose an action using an epsilon-greedy policy.

        :param state: The current state of the Boolean network.
        :return: The chosen action.
        """
        possible_actions = self.get_possible_actions(state)
        if np.random.rand() < self.epsilon:
            # Explore: choose a random action
            return np.random.choice(possible_actions)
        else:
            # Exploit: choose the action with the highest Q-value
            state_key = self.get_state_key(state)
            q_values = [self.q_table.get((state_key, action), 0) for action in possible_actions]
            return possible_actions[np.argmax(q_values)]

    def get_possible_actions(self, state):
        """
        Define possible actions based on the current state.

        :param state: The current state of the Boolean network.
        :return: A list of possible actions.
        """
        # Nodes with a value of None – we will generate all combinations for these nodes
        none_nodes = [node for node in nodes if initial_values.get(node) is None]

        # All possible combinations of 0/1 for the nodes that are None
        possible_replace_none = list(product([0, 1], repeat=len(none_nodes)))
        possible_actions = []
        visited_states = set()
        for condition in possible_replace_none:
            # Build a new initial state for the current combination
            possible_conditions = initial_values.copy()
            for idx, node in enumerate(none_nodes):
                possible_conditions[node] = condition[idx]

            # Ensure we haven't already visited this state
            trial_state_key = tuple(sorted(possible_conditions.items()))
            if trial_state_key in visited_states:
                continue
            visited_states.add(trial_state_key)
            possible_actions.append(possible_conditions)
        # For simplicity, let's assume actions are flipping the value of a node
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
        for node, target_value in self.target_values.items():
            if next_state[node] == target_value:
                reward += 1
        return reward

    def update_q_table(self, state, action, reward, next_state):
        """
        Update the Q-table using the Q-learning update rule.

        :param state: The current state.
        :param action: The action taken.
        :param reward: The reward received.
        :param next_state: The next state.
        """
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)

        # Get the current Q-value - Q(s,a):
        current_q = self.q_table.get((state_key, action), 0)

        # Get the maximum Q-value for the next state- max_a Q(s',a):
        possible_actions = self.get_possible_actions(next_state)
        max_next_q = max([self.q_table.get((next_state_key, a), 0) for a in possible_actions])

        # Update the Q-value using the Q-learning formula
        # new Q(s,a) <- Q(s,a) + alpha [R + gamma * max_a Q(s',a)-Q(s,a)]
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.q_table[(state_key, action)] = new_q

    def train(self, episodes, initial_values = initial_values):
        """
        Train the Q-learning agent over a number of episodes.

        :param episodes: The number of episodes to train.
        """
        for episode in range(episodes):
            state = self.get_initial_state(initial_values = initial_values)  # Get the initial state of the network
            done = False

            # Choose an action
            action = self.choose_action(state)
            state = action
            i = 0
            while not done & i<10:
                # Take the action and observe the next state and reward
                next_state = evaluate_state(state, self.primes, self.edge_functions)
                reward = self.get_reward(state, action, next_state)

                # Update the Q-table
                self.update_q_table(state, action, reward, next_state)

                # Transition to the next state
                state = next_state

                # Check if the episode is done
                done = self.is_terminal_state(state)
                i += 1

    def get_initial_state(self, initial_values):
        """
        Define the initial state of the network.

        :return: The initial state.
        """
        return {node: v for node, v in initial_values.items()}

    def is_terminal_state(self, state):
        """
        Check if the state is terminal (i.e., matches the target values).

        :param state: The current state.
        :return: True if the state is terminal, False otherwise.
        """
        return all(state[node] == value for node, value in self.target_values.items())

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
