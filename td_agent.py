"""
Temporal Difference (TD) Learning Agent for Boolean Network Control.

This module implements a TD learning agent that learns to control a Boolean network
to reach target states. The agent uses value function approximation, experience replay,
state aggregation, and intelligent action sampling to efficiently learn optimal policies.
"""

import numpy as np
from itertools import product
import logging
import sqlite3


class TDAgent:
    """
    A Temporal Difference learning agent for Boolean network control.
    
    The agent learns optimal control strategies by estimating state values and using
    epsilon-greedy exploration with intelligent action selection. It employs various
    optimizations including state aggregation, experience replay, and heuristic guidance
    to handle large state spaces efficiently.
    """
    def __init__(self, primes, edge_functions, target_values, initial_values, alpha=0.1, gamma=0.9, epsilon=0.1, 
                 use_state_aggregation=True, aggregation_level=0.8):
        """
        Initialize the TD learning agent.

        :param primes: A dictionary of node functions (primes).
        :param edge_functions: A dictionary of edge functions.
        :param target_values: A dictionary of target values for each node.
        :param alpha: Learning rate (default: 0.1).
        :param gamma: Discount factor (default: 0.9).
        :param epsilon: Exploration rate (default: 0.1).
        :param use_state_aggregation: Whether to use state aggregation (default: True).
        :param aggregation_level: Level of state aggregation 0-1 (default: 0.8).
        """
        self.primes = primes
        self.edge_functions = edge_functions
        self.target_values = target_values
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.use_state_aggregation = use_state_aggregation
        self.aggregation_level = aggregation_level
        self.initial_values = dict(initial_values)  # Convert to dictionary if it isn't
        self.current_state = initial_values.copy()  # Initialize current state
        self.success = False

        # Identify controllable nodes (those with None in initial_values)
        self.controllable_nodes = [node for node, val in initial_values.items() if val is None]
        self.fixed_nodes = {node: val for node, val in initial_values.items() if val is not None}
        
        # Don't pre-generate all actions - use lazy generation instead
        self.action_space_size = 2 ** len(self.controllable_nodes) if len(self.controllable_nodes) <= 20 else float('inf')
        self.max_actions_to_evaluate = min(1000, self.action_space_size)  # Limit action evaluation

        # Initialize value table as empty dictionary (lazy initialization)
        self.value_table = {}
        self.default_value = 0.0  # Default value for unseen states
        self.db_path = "td_agent_value_table.db"
        self._init_db()
        
        # Experience replay buffer
        self.experience_buffer = []
        self.buffer_size = 10000  # Maximum number of experiences to store
        self.batch_size = 32  # Number of experiences to sample for replay
        self.replay_frequency = 4  # Replay every N steps
        
        # Adaptive learning parameters
        self.learning_stats = {
            'episodes_completed': 0,
            'successful_episodes': 0,
            'total_steps': 0,
            'value_table_size': 0
        }
        
        # Memory management
        self.max_value_table_size = 50000  # Maximum value table entries
        self.cleanup_frequency = 100  # Clean up value table every N episodes

    def get_state_value(self, state_key):
        """
        Get state value for a state key with lazy initialization and approximation.
        
        First checks the in-memory cache, then the database. If the state is not found,
        approximates its value using heuristics and similar states.
        
        :param state_key: Hashable key representing the state (tuple or dict).
        :return: Estimated value for the state.
        """
        if state_key not in self.value_table:
            db_value = self._get_db_value(state_key)
            if db_value is not None:
                self.value_table[state_key] = db_value
            else:
                state_dict = dict(state_key) if isinstance(state_key, tuple) else state_key
                approximated_value = self.approximate_state_value(state_dict)
                self.value_table[state_key] = approximated_value
                self._set_db_value(state_key, approximated_value)
        return self.value_table[state_key]
    
    def aggregate_state(self, state):
        """
        Aggregate similar states to reduce state space complexity.
        
        :param state: The original state dictionary.
        :return: An aggregated state representation.
        """
        if not self.use_state_aggregation:
            return state
        
        # Create aggregated state by grouping similar nodes
        aggregated = {}
        node_groups = {}
        
        # Group nodes by their importance (target nodes get priority)
        for node in state:
            if node in self.target_values:
                # Target nodes are kept as-is (high precision)
                aggregated[node] = state[node]
            else:
                # Non-target nodes can be aggregated
                # Use a simple hash-based grouping
                group_id = hash(node) % max(1, int(len(self.primes) * self.aggregation_level))
                if group_id not in node_groups:
                    node_groups[group_id] = []
                node_groups[group_id].append(node)
        
        # Aggregate non-target nodes
        for group_id, nodes in node_groups.items():
            if len(nodes) > 1:
                # Use majority vote or average for the group
                values = [state[node] for node in nodes]
                aggregated[f"group_{group_id}"] = int(sum(values) / len(values) > 0.5)
            else:
                # Single node in group, keep as-is
                aggregated[nodes[0]] = state[nodes[0]]
        
        return aggregated

    # ============================================================================
    # Experience Replay Methods
    # ============================================================================

    def store_experience(self, state, action, reward, next_state, done):
        """
        Store an experience tuple in the replay buffer.
        
        :param state: Current state
        :param action: Action taken
        :param reward: Reward received
        :param next_state: Next state
        :param done: Whether episode is done
        """
        experience = (state, action, reward, next_state, done)
        self.experience_buffer.append(experience)
        
        # Maintain buffer size
        if len(self.experience_buffer) > self.buffer_size:
            self.experience_buffer.pop(0)
    
    def sample_experiences(self):
        """
        Sample a batch of experiences from the replay buffer.
        
        :return: List of experience tuples
        """
        if len(self.experience_buffer) < self.batch_size:
            return self.experience_buffer
        
        # Random sampling
        indices = np.random.choice(len(self.experience_buffer), self.batch_size, replace=False)
        return [self.experience_buffer[i] for i in indices]
    
    def replay_experiences(self):
        """
        Perform experience replay to improve learning efficiency.
        """
        if len(self.experience_buffer) < self.batch_size:
            return
        
        experiences = self.sample_experiences()
        
        for state, action, reward, next_state, done in experiences:
            # Update value table using the stored experience
            state_key = self.get_state_key(state)
            next_state_key = self.get_state_key(next_state)
            
            current_value = self.get_state_value(state_key)
            next_value = self.get_state_value(next_state_key) if not done else 0.0
            
            # TD learning update
            new_value = current_value + self.alpha * (reward + self.gamma * next_value - current_value)
            self.value_table[state_key] = new_value

    # ============================================================================
    # Memory Management Methods
    # ============================================================================
    
    def cleanup_value_table(self):
        """
        Clean up value table by removing low-value entries to manage memory.
        """
        if len(self.value_table) <= self.max_value_table_size:
            return

        # Sort state values and keep only the top entries
        sorted_entries = sorted(self.value_table.items(), key=lambda x: abs(x[1]), reverse=True)

        # Keep top entries in memory, others only in DB
        self.value_table = dict(sorted_entries[:self.max_value_table_size])
        self._sync_cache_to_db()
        self.learning_stats['value_table_size'] = len(self.value_table)
        logging.info(f"Value table cleaned up. Current size: {len(self.value_table)}")
    
    def adaptive_epsilon_decay(self):
        """
        Adaptively decay epsilon based on learning progress.
        """
        success_rate = self.learning_stats['successful_episodes'] / max(1, self.learning_stats['episodes_completed'])
        
        # Decay epsilon faster if we're learning well
        if success_rate > 0.3:  # If success rate is good
            self.epsilon = max(0.01, self.epsilon * 0.995)
        else:
            self.epsilon = min(0.5, self.epsilon * 1.001)  # Increase exploration if struggling

    # ============================================================================
    # Statistics and Monitoring Methods
    # ============================================================================
    
    def get_learning_stats(self):
        """
        Get current learning statistics.
        
        :return: Dictionary of learning statistics
        """
        self.learning_stats['value_table_size'] = len(self.value_table)
        return self.learning_stats.copy()

    # ============================================================================
    # Heuristic and Action Selection Methods
    # ============================================================================
    
    def heuristic_value(self, state):
        """
        Fast heuristic evaluation of state quality without full simulation.
        
        :param state: The state to evaluate
        :return: Heuristic value (higher = better)
        """
        score = 0.0
        
        # Direct target matching (highest priority)
        for node, target in self.target_values.items():
            if state.get(node) == target:
                score += 10.0
        
        # Partial progress evaluation
        for node, target in self.target_values.items():
            if node in state:
                if state[node] == target:
                    score += 5.0
                else:
                    # Small penalty for incorrect values
                    score -= 1.0
        
        # Stability bonus: reward states that don't change (avoid oscillations)
        if hasattr(self, 'previous_state') and state == self.previous_state:
            score += 2.0
        
        # Bonus for controllable nodes that match their targets
        controllable_correct = sum(1 for node in self.controllable_nodes 
                                 if node in self.target_values and 
                                 state.get(node) == self.target_values[node])
        score += controllable_correct * 3.0
        
        return score
    
    def get_promising_actions(self, state, top_k=5):
        """
        Get the most promising actions based on intelligent sampling and evaluation.
        
        :param state: Current state
        :param top_k: Number of top actions to return
        :return: List of most promising actions
        """
        # Intelligently sample actions instead of evaluating all possibilities
        num_samples = min(self.max_actions_to_evaluate, max(top_k * 5, 20))
        sampled_actions = self.sample_actions_intelligently(state, num_samples)
        
        action_scores = []
        
        for action in sampled_actions:
            # Evaluate using heuristic function
            heuristic_score = self.heuristic_value(action)
            
            # Get learned state value from value table
            action_key = self.get_state_key(action)
            state_value = self.get_state_value(action_key)
            
            # Combine heuristic and learned value (70% heuristic, 30% learned)
            combined_score = 0.7 * heuristic_score + 0.3 * state_value
            action_scores.append((combined_score, action))
        
        # Sort by combined score and return top-k actions
        action_scores.sort(reverse=True, key=lambda x: x[0])
        return [action for _, action in action_scores[:top_k]]
    
    def sample_actions_intelligently(self, state, num_samples=100):
        """
        Sample actions intelligently without enumerating all possibilities.
        
        :param state: Current state
        :param num_samples: Number of actions to sample
        :return: List of sampled actions
        """
        if len(self.controllable_nodes) == 0:
            return [state.copy()]
        
        sampled_actions = []
        
        # Strategy 1: Target-directed sampling (prioritize actions that set target nodes)
        target_actions = []
        for node in self.controllable_nodes:
            if node in self.target_values:
                # Create action that sets this controllable node to its target value
                action = state.copy()
                action[node] = self.target_values[node]
                target_actions.append(action)
        
        sampled_actions.extend(target_actions[:num_samples//3])
        
        # Strategy 2: Biased random sampling
        # For target nodes: 70% chance to set to target, 30% random
        # For non-target nodes: completely random
        remaining_samples = num_samples - len(sampled_actions)
        for _ in range(remaining_samples):
            action = state.copy()
            for node in self.controllable_nodes:
                if node in self.target_values:
                    # Bias toward target value
                    action[node] = self.target_values[node] if np.random.rand() < 0.7 else np.random.randint(0, 2)
                else:
                    # Random value for non-target nodes
                    action[node] = np.random.randint(0, 2)
            sampled_actions.append(action)
        
        # Remove duplicate actions
        unique_actions = []
        seen = set()
        for action in sampled_actions:
            action_key = tuple(sorted(action.items()))
            if action_key not in seen:
                seen.add(action_key)
                unique_actions.append(action)
        
        return unique_actions[:num_samples]
    
    def should_prune_path(self, state, depth=0, max_depth=10):
        """
        Determine if a path should be pruned early to save computation.
        
        :param state: Current state
        :param depth: Current search depth
        :param max_depth: Maximum allowed depth
        :return: True if path should be pruned
        """
        # Prune if too deep
        if depth > max_depth:
            return True
        
        # Prune if heuristic value is too low
        heuristic_val = self.heuristic_value(state)
        if heuristic_val < 0:  # Very unpromising states
            return True
        
        # Prune if we've seen this state too many times (avoid loops)
        state_key = self.get_state_key(state)
        if hasattr(self, 'state_visit_count'):
            if self.state_visit_count.get(state_key, 0) > 3:
                return True
        
        return False
    
    def update_state_visit_count(self, state):
        """
        Track how many times we've visited each state.
        
        Used for path pruning to avoid getting stuck in loops.
        
        :param state: The state to update the visit count for.
        """
        if not hasattr(self, 'state_visit_count'):
            self.state_visit_count = {}
        
        state_key = self.get_state_key(state)
        self.state_visit_count[state_key] = self.state_visit_count.get(state_key, 0) + 1
    
    def find_shortest_path_to_goal(self, current_state, max_steps=20):
        """
        Use goal-directed search to find shortest path to target.
        
        :param current_state: Starting state
        :param max_steps: Maximum steps to search
        :return: Shortest path if found, None otherwise
        """
        from collections import deque
        
        # BFS with heuristic guidance
        queue = deque([(current_state, [])])
        visited = set()
        
        for step in range(max_steps):
            if not queue:
                break
                
            state, path = queue.popleft()
            state_key = self.get_state_key(state)
            
            if state_key in visited:
                continue
            visited.add(state_key)
            
            # Check if goal reached
            if self.is_terminal_state(state):
                return path + [state]
            
            # Get promising next states
            promising_actions = self.get_promising_actions(state, top_k=3)
            
            for action in promising_actions:
                if not self.should_prune_path(action, len(path)):
                    next_state = self.evaluate_state(action)
                    queue.append((next_state, path + [action]))
        
        return None
    
    def approximate_state_value(self, state):
        """
        Fast approximation of state value for unseen states using heuristics and similarity.
        
        :param state: State to approximate state value for
        :return: Approximated state value
        """
        # Base heuristic value
        heuristic_val = self.heuristic_value(state)
        
        # Find similar states in value table
        similar_values = []
        state_key = self.get_state_key(state)
        
        for stored_key, value in self.value_table.items():
            # Calculate similarity (number of matching node values)
            if isinstance(stored_key, tuple) and isinstance(state_key, tuple):
                stored_dict = dict(stored_key)
                current_dict = dict(state_key)
                
                # Count matching values
                matches = sum(1 for k in current_dict if k in stored_dict and current_dict[k] == stored_dict[k])
                total = len(current_dict)
                similarity = matches / total if total > 0 else 0
                
                if similarity > 0.7:  # High similarity threshold
                    similar_values.append((similarity, value))
        
        # Combine heuristic and similar states
        if similar_values:
            # Weight by similarity
            weighted_sum = sum(sim * value for sim, value in similar_values)
            total_weight = sum(sim for sim, _ in similar_values)
            similar_avg = weighted_sum / total_weight if total_weight > 0 else 0
            
            # Combine: 60% heuristic, 40% similar states
            return 0.6 * heuristic_val + 0.4 * similar_avg
        else:
            # No similar states found, use pure heuristic
            return heuristic_val * 0.1  # Scale down heuristic for state value range

    def get_state_key(self, state):
        """
        Convert the state to a hashable key (e.g., tuple) for the value table.

        :param state: The current state of the Boolean network.
        :return: A hashable key representing the state.
        """
            # Use aggregated state if enabled
        if self.use_state_aggregation:
            aggregated_state = self.aggregate_state(state)
            return tuple(sorted(aggregated_state.items()))
        else:
            # Convert state dictionary to immutable tuple key
            return tuple((k, state[k]) for k in self.primes)

    # ============================================================================
    # State and Action Representation Methods
    # ============================================================================

    def get_action_key(self, action):
        """
        Convert the action to a hashable key (e.g., tuple) for the value table.

        :param action: The action taken.
        :return: A hashable key representing the action.
        """
        return tuple(sorted((k, v) for k, v in action.items()))

    def choose_action(self):
        """
        Choose an action using intelligent epsilon-greedy policy with heuristic guidance.

        Uses epsilon-greedy exploration: with probability epsilon, explores by selecting
        from promising actions. Otherwise, exploits by selecting the action with the
        highest estimated value from the promising set.

        :return: The chosen action dictionary.
        """
        # Get promising actions (intelligently sampled)
        promising_actions = self.get_promising_actions(self.current_state, top_k=10)
        
        # Epsilon-greedy action selection
        if np.random.rand() < self.epsilon:
            # Explore: choose from promising actions (guided exploration)
            if len(promising_actions) > 0:
                return np.random.choice(promising_actions)
            else:
                # Fallback to intelligent sampling if no promising actions
                fallback_actions = self.sample_actions_intelligently(self.current_state, 10)
                return np.random.choice(fallback_actions) if fallback_actions else self.current_state
        else:
            # Exploit: choose best action from promising set
            if len(promising_actions) > 0:
                # Evaluate state values for promising actions only
                best_action = None
                best_value = float('-inf')
                
                for action in promising_actions:
                    try:
                        action_key = self.get_state_key(action)
                        state_value = self.get_state_value(action_key)
                        if state_value > best_value:
                            best_value = state_value
                            best_action = action
                    except Exception as e:
                        logging.error(f"Error processing action {action}: {str(e)}")
                        continue
                
                return best_action if best_action is not None else promising_actions[0]
            else:
                # Fallback to intelligent sampling
                fallback_actions = self.sample_actions_intelligently(self.current_state, 5)
                return fallback_actions[0] if fallback_actions else self.current_state

    def get_possible_actions(self, state=None):
        """
        Generate all possible actions based on the current state.

        Creates all possible combinations of values for nodes that have None values
        in the current state.

        :param state: The current state of the Boolean network (defaults to current_state).
        :return: A list of all possible action dictionaries.
        """
        if state is None:
            state = self.current_state

        # Find nodes with None values - these can be controlled
        none_nodes = [node for node in state if state.get(node) is None]

        # Generate all possible combinations of 0/1 for the controllable nodes
        possible_replace_none = list(product([0, 1], repeat=len(none_nodes)))
        possible_actions = []
        visited_states = set()
        
        for condition in possible_replace_none:
            # Build a new state by setting controllable nodes to this combination
            possible_conditions = state.copy()
            for idx, node in enumerate(none_nodes):
                possible_conditions[node] = condition[idx]

            # Avoid duplicate states
            trial_state_key = tuple(sorted(possible_conditions.items()))
            if trial_state_key in visited_states:
                continue
            visited_states.add(trial_state_key)
            possible_actions.append(possible_conditions)
        
        return possible_actions

    def evaluate_state(self, state):
        """
        Evaluate the next state of the network according to the primes and edge functions.
        
        Applies node functions (primes) first, then edge functions to compute the next
        state of the Boolean network from the current state.
        
        :param state: The current state dictionary.
        :return: The next state dictionary after applying all network functions.
        """
        new_state = state.copy()

        # Evaluate node functions (primes)
        for node, func in self.primes.items():
            if isinstance(func, str):
                # Function is a string expression - evaluate it
                try:
                    new_state[node] = eval(func, {}, new_state)
                except Exception as e:
                    logging.warning(f"Error evaluating function for node {node}: {e}")
                    new_state[node] = 0  # Default to 0 on error
            elif callable(func):
                # Function is a callable - execute it
                new_state[node] = func(state)
            else:
                # Function is a constant value - keep current state
                new_state[node] = state[node]

        # Evaluate edge functions (between nodes)
        for (start_node, end_node), edge_func in self.edge_functions.items():
            if start_node in state:
                try:
                    new_state[end_node] = eval(edge_func, {}, new_state)
                except Exception as e:
                    logging.warning(f"Error evaluating edge function ({start_node}, {end_node}): {e}")
                    new_state[end_node] = 0  # Default to 0 on error

        return new_state
    

    # ============================================================================
    # Reward and Value Update Methods
    # ============================================================================

    def get_reward(self, state, next_state, step_penalty=0):
        """
        Calculate the reward based on the target values and distance to goal.

        Returns a high positive reward for reaching the target state, or a
        reward proportional to progress toward the target for non-terminal states.

        :param state: The current state.
        :param next_state: The next state.
        :param step_penalty: Penalty for number of steps taken (default: 0).
        :return: The calculated reward value.
        """
        if self.is_terminal_state(next_state):
            # Terminal state reached: give base reward minus step penalty
            # Shorter paths receive higher rewards
            base_reward = 10.0
            reward = base_reward - (step_penalty * 0.1)
        else:
            # Non-terminal state: reward based on progress toward target
            # Calculate what fraction of target nodes match their target values
            matching_nodes = sum(1 for node, target in self.target_values.items() 
                               if next_state.get(node) == target)
            total_nodes = len(self.target_values)
            distance_reward = (matching_nodes / total_nodes) * 2.0
            # Small negative base to encourage continued progress
            reward = distance_reward - 0.5
        return reward

    def update_value_table(self, state, reward, next_state):
        """
        Update the value table using the TD learning update rule.

        Applies the standard TD(0) update: V(s) <- V(s) + alpha [R + gamma * V(s') - V(s)]
        Updates both the in-memory cache and persistent database storage.

        :param state: The current state.
        :param reward: The reward received for the transition.
        :param next_state: The next state reached.
        """
        # Convert states to hashable keys
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)

        # Get current and next state values (initialize if not present)
        current_value = self.get_state_value(state_key)
        next_state_value = self.get_state_value(next_state_key)

        # Apply TD learning update rule: V(s) <- V(s) + alpha [R + gamma * V(s') - V(s)]
        new_value = current_value + self.alpha * (reward + self.gamma * next_state_value - current_value)
        
        # Update both in-memory cache and database
        self.value_table[state_key] = new_value
        self._set_db_value(state_key, new_value)


    # ============================================================================
    # Main Training and Control Methods
    # ============================================================================

    def train(self, episodes):
        """
        Train the TD learning agent over a specified number of episodes.

        Each episode consists of:
        1. Choosing an action using epsilon-greedy policy
        2. Simulating the network dynamics
        3. Updating value estimates using TD learning
        4. Performing experience replay periodically
        5. Applying final rewards based on success/failure

        :param episodes: The number of training episodes to execute.
        """
        for episode in range(episodes):
            done = False
            self.learning_stats['episodes_completed'] += 1
            
            # Adaptive epsilon decay
            self.adaptive_epsilon_decay()
            
            # Periodic cleanup
            if episode % self.cleanup_frequency == 0:
                self.cleanup_value_table()
            
            # Choose an action using epsilon-greedy policy
            action = self.choose_action()
            
            # Check if agent chose to stay in current state (indicating completion)
            if action == self.current_state:
                if self.success:
                    logging.info(f"Simulation completed: reached target state in {self.learning_stats['total_steps']} steps")
                else:
                    logging.info("Simulation completed: no action led to the target state")
                break
            
            # Transition to the chosen action state
            self.current_state = action
            steps = 0
            intermediate_states = [action]
            while not done and steps < 200:  # Limit steps to avoid infinite loops
                # Update state visit count for pruning
                self.update_state_visit_count(self.current_state)
                
                # Early termination: try to find direct path to goal
                if steps == 0:  # Only check at the beginning of episode
                    direct_path = self.find_shortest_path_to_goal(self.current_state, max_steps=10)
                    if direct_path:
                        # Found a short path, use it
                        intermediate_states.extend(direct_path[:-1])  # All but the last state
                        self.current_state = direct_path[-1]  # Set to final state
                        steps += len(direct_path)
                        done = True
                        break
                
                # Evaluate the next state using the Boolean network update rules
                next_state = self.evaluate_state(self.current_state)
                
                # Early pruning: skip unpromising states to save computation
                if self.should_prune_path(next_state, steps):
                    done = True
                    break
                
                # Store the intermediate state for later reward assignment
                intermediate_states.append(next_state.copy())
                
                # Check if the next state has potential to lead to target
                next_state_key = self.get_state_key(next_state)
                if self.get_state_value(next_state_key) >= 0.0:
                    reward = self.get_reward(self.current_state, next_state, steps)

                    # Store experience for replay
                    self.store_experience(self.current_state, None, reward, next_state, False)

                    # Update the value table
                    self.update_value_table(self.current_state, reward, next_state)
                    
                    # Perform experience replay periodically
                    if steps % self.replay_frequency == 0:
                        self.replay_experiences()

                    # Check if the state has stabilized
                    done = self.current_state == next_state

                    # Transition to the next state
                    self.current_state = next_state
                    steps += 1
                else:
                    # Stop if the next state cannot lead to target
                    done = True

            # Apply final rewards based on episode outcome (success or failure)
            if self.is_terminal_state(self.current_state):
                # SUCCESS: Assign positive rewards with step-based penalty
                # Shorter paths receive higher rewards
                self.learning_stats['successful_episodes'] += 1
                base_success_reward = 10.0
                step_penalty = len(intermediate_states) * 0.1
                final_reward = base_success_reward - step_penalty
                self.success = True
                
                # Update intermediate states with success rewards
                # States closer to goal receive higher rewards
                for i in range(len(intermediate_states)):
                    current_state = intermediate_states[i]
                    if i < len(intermediate_states) - 1:
                        next_state = intermediate_states[i + 1]
                    else:
                        next_state = self.current_state
                    
                    # Reward decreases with distance from goal
                    distance_from_goal = len(intermediate_states) - i - 1
                    state_reward = final_reward - (distance_from_goal * 0.5)
                    self.update_value_table(current_state, state_reward, next_state)
            else:
                # FAILURE: Assign negative rewards to discourage these paths
                failure_penalty = -5.0
                self.success = False
                
                # Update intermediate states with negative rewards
                for i in range(len(intermediate_states)):
                    current_state = intermediate_states[i]
                    if i < len(intermediate_states) - 1:
                        next_state = intermediate_states[i + 1]
                    else:
                        next_state = self.current_state
                    
                    # Apply failure penalty to discourage these paths
                    self.update_value_table(current_state, failure_penalty, next_state)
            
            # Reset to the action that started this episode
            self.current_state = action.copy()
            
            # Update total steps
            self.learning_stats['total_steps'] += steps
            
        if not done:
            print("not found")
        
        # Final cleanup and stats
        self.cleanup_value_table()
        logging.info(f"Training completed. Stats: {self.get_learning_stats()}")


    # ============================================================================
    # State Query Methods
    # ============================================================================

    def get_initial_state(self):
        """
        Get the initial state of the network.

        :return: A dictionary representing the initial state.
        """
        return {node: v for node, v in self.initial_values.items()}

    def is_terminal_state(self, state=None):
        """
        Check if the state is terminal (i.e., matches all target values).

        :param state: The state to check (defaults to current_state).
        :return: True if the state matches all target values, False otherwise.
        """
        if state is None:
            state = self.current_state

        terminal = all(state.get(node, None) == target for node, target in self.target_values.items())
        return terminal

    def get_optimal_policy(self):
        """
        Extract the optimal policy from the value table.

        For each state, selects the action that leads to the highest value.
        Note: This implementation assumes state-action pairs are stored in
        the value table, which may need adjustment based on actual storage format.

        :return: A dictionary mapping states to their optimal actions.
        """
        policy = {}
        for (state_key, action), state_value in self.value_table.items():
            if state_key not in policy or state_value > self.value_table.get((state_key, policy[state_key]), 0):
                policy[state_key] = action
        return policy

    # ============================================================================
    # Database and Persistence Methods
    # ============================================================================

    def _init_db(self):
        """
        Initialize the SQLite database for persistent value table storage.
        
        Creates a table to store state keys and their corresponding values if it
        doesn't already exist.
        """
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        self.cursor.execute(
            "CREATE TABLE IF NOT EXISTS value_table (state_key TEXT PRIMARY KEY, value REAL)"
        )
        self.conn.commit()

    def _serialize_key(self, state_key):
        """
        Convert a state key (tuple) to a string for database storage.
        
        :param state_key: The hashable state key to serialize.
        :return: String representation of the key.
        """
        return str(state_key)

    def _deserialize_key(self, key_str):
        """
        Convert a string back to a tuple state key.
        
        :param key_str: The string representation of the key.
        :return: The original state key tuple.
        """
        import ast
        return ast.literal_eval(key_str)

    def _get_db_value(self, state_key):
        """
        Retrieve a state value from the database.
        
        :param state_key: The state key to look up.
        :return: The stored value if found, None otherwise.
        """
        key_str = self._serialize_key(state_key)
        self.cursor.execute("SELECT value FROM value_table WHERE state_key=?", (key_str,))
        row = self.cursor.fetchone()
        return row[0] if row else None

    def _set_db_value(self, state_key, value):
        """
        Store or update a state value in the database.
        
        :param state_key: The state key to store.
        :param value: The value to associate with the key.
        """
        key_str = self._serialize_key(state_key)
        self.cursor.execute(
            "INSERT OR REPLACE INTO value_table (state_key, value) VALUES (?, ?)", (key_str, value)
        )
        self.conn.commit()

    def _sync_cache_to_db(self):
        """
        Synchronize all entries from the in-memory cache to the database.
        
        Used during cleanup to persist values before removing them from memory.
        """
        for state_key, value in self.value_table.items():
            self._set_db_value(state_key, value)
