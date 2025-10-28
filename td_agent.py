import numpy as np
from itertools import product
import logging
import sqlite3



class TDAgent:
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

        #logging.debug(f"Initial value table: {self.value_table}")  # Debug

    def get_state_value(self, state_key):
        """Get state value for a state, with lazy initialization and approximation"""
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
    
    def get_learning_stats(self):
        """
        Get current learning statistics.
        
        :return: Dictionary of learning statistics
        """
        self.learning_stats['value_table_size'] = len(self.value_table)
        return self.learning_stats.copy()
    
    def heuristic_value(self, state):
        """
        Fast heuristic evaluation of state quality without full simulation.
        
        :param state: The state to evaluate
        :return: Heuristic value (higher = better)
        """
        score = 0.0
        
        # Direct target matching (most important)
        for node, target in self.target_values.items():
            if state.get(node) == target:
                score += 10.0  # High reward for correct target values
        
        # Partial progress (nodes moving toward target)
        for node, target in self.target_values.items():
            if node in state:
                # Reward being closer to target (even if not exact)
                if state[node] == target:
                    score += 5.0
                else:
                    # Small penalty for wrong values
                    score -= 1.0
        
        # Stability bonus (avoid oscillating states)
        if hasattr(self, 'previous_state') and state == self.previous_state:
            score += 2.0
        
        # Controllable node optimization
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
        # Sample actions intelligently instead of evaluating all
        num_samples = min(self.max_actions_to_evaluate, max(top_k * 5, 20))
        sampled_actions = self.sample_actions_intelligently(state, num_samples)
        
        action_scores = []
        
        for action in sampled_actions:
            # Quick heuristic evaluation
            heuristic_score = self.heuristic_value(action)
            
            # Combine with state value if available
            action_key = self.get_state_key(action)
            state_value = self.get_state_value(action_key)
            
            # Weighted combination: 70% heuristic, 30% state value
            combined_score = 0.7 * heuristic_score + 0.3 * state_value
            action_scores.append((combined_score, action))
        
        # Sort by score and return top-k
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
        
        # Strategy 1: Target-directed sampling (high priority)
        target_actions = []
        for node in self.controllable_nodes:
            if node in self.target_values:
                # Create action that sets this node to target value
                action = state.copy()
                action[node] = self.target_values[node]
                target_actions.append(action)
        
        sampled_actions.extend(target_actions[:num_samples//3])
        
        # Strategy 2: Random sampling with bias toward promising values
        remaining_samples = num_samples - len(sampled_actions)
        for _ in range(remaining_samples):
            action = state.copy()
            for node in self.controllable_nodes:
                if node in self.target_values:
                    # 70% chance to set to target value
                    action[node] = self.target_values[node] if np.random.rand() < 0.7 else np.random.randint(0, 2)
                else:
                    # Random for non-target nodes
                    action[node] = np.random.randint(0, 2)
            sampled_actions.append(action)
        
        # Remove duplicates
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
        """Track how many times we've visited each state"""
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
            return tuple((k, state[k]) for k in self.primes)  # Convert state dictionary to immutable tuple

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

        :return: The chosen action.
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

    def update_value_table(self, state, reward, next_state):
        """
        Update the value table using the TD learning update rule.

        :param state: The current state.
        :param reward: The reward received.
        :param next_state: The next state.
        """
        # Convert states and actions to hashable keys
        state_key = self.get_state_key(state)  # Use the passed state, not current_state
        # action_key = self.get_action_key(action)
        next_state_key = self.get_state_key(next_state)

        # Debugging print
        #logging.info("\n--- Updating value table ---")  # Debug
        #print(f"State: {state}, Action: {action}, Reward: {reward}, Next State: {next_state}")

        # Get current state value (initialize if not present) - V(s):
        current_value = self.get_state_value(state_key)

        next_state_value = self.get_state_value(next_state_key)

        # TD learning update
        # new V(s) <- V(s) + alpha [R + gamma * V(s')-V(s)]
        new_value = current_value + self.alpha * (reward + self.gamma * next_state_value - current_value)
        self.value_table[(state_key)] = new_value
        self._set_db_value(state_key, new_value)
        logging.debug(f"\n Value table in state {state_key} updated to: {new_value}")  # Debug


    def train(self, episodes):
        """
        Train the TD learning agent over a number of episodes.

        :param episodes: The number of episodes to train.
        """
        for episode in range(episodes):
            done = False
            self.learning_stats['episodes_completed'] += 1
            
            # Adaptive epsilon decay
            self.adaptive_epsilon_decay()
            
            # Periodic cleanup
            if episode % self.cleanup_frequency == 0:
                self.cleanup_value_table()
            
            # Choose an action
            action = self.choose_action()
            if action == self.current_state:
                if self.success:
                    logging.info(f"\n Simulation completed: action {action} reached the target state in {self.learning_stats['total_steps']} steps")
                else:
                    logging.info(f"\n Simulation completed: no action led to the target state")
                break    
            self.current_state = action     # Use the action to set the current state
            logging.debug(f"\n action checks: {action}")  # Debug
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
                
                # Early pruning: skip unpromising states
                if self.should_prune_path(next_state, steps):
                    done = True
                    break
                
                # Store the intermediate state
                intermediate_states.append(next_state.copy())
                
                next_state_key = self.get_state_key(next_state)
                if self.get_state_value(next_state_key) >= 0.0: # If the next state can lead to target
                    reward = self.get_reward(self.current_state, next_state, steps)

                    # Store experience for replay
                    self.store_experience(self.current_state, None, reward, next_state, False)

                    # Update the value table
                    self.update_value_table(self.current_state, reward, next_state)
                    
                    # Perform experience replay periodically
                    if steps % self.replay_frequency == 0:
                        self.replay_experiences()

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
                self.learning_stats['successful_episodes'] += 1
                base_success_reward = 10.0
                step_penalty = len(intermediate_states) * 0.1
                final_reward = base_success_reward - step_penalty
                self.success = True
                
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
                    self.update_value_table(current_state, state_reward, next_state)
                    
            else:
                # FAILURE: Give negative rewards to discourage these paths
                failure_penalty = -5.0
                self.success = False
                
                # Update intermediate states with negative rewards
                for i in range(len(intermediate_states)):
                    current_state = intermediate_states[i]
                    if i < len(intermediate_states) - 1:
                        next_state = intermediate_states[i + 1]
                    else:
                        next_state = self.current_state
                    
                    # Apply failure penalty to discourage these intermediate states
                    self.update_value_table(current_state, failure_penalty, next_state)
            self.current_state= action.copy()  # Store the last action taken
            
            # Update total steps
            self.learning_stats['total_steps'] += steps
            
        if not done:
            print("not found")
        
        # Final cleanup and stats
        self.cleanup_value_table()
        logging.info(f"Training completed. Stats: {self.get_learning_stats()}")


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
        Extract the optimal policy from the value table.

        :return: A dictionary mapping states to optimal actions.
        """
        policy = {}
        for (state_key, action), state_value in self.value_table.items():
            if state_key not in policy or state_value > self.value_table.get((state_key, policy[state_key]), 0):
                policy[state_key] = action
        return policy

    # --- Hybrid cache and SQLite DB methods ---

    def _init_db(self):
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        self.cursor.execute(
            "CREATE TABLE IF NOT EXISTS value_table (state_key TEXT PRIMARY KEY, value REAL)"
        )
        self.conn.commit()

    def _serialize_key(self, state_key):
        # Convert tuple to string for DB storage
        return str(state_key)

    def _deserialize_key(self, key_str):
        # Convert string back to tuple
        import ast
        return ast.literal_eval(key_str)

    def _get_db_value(self, state_key):
        key_str = self._serialize_key(state_key)
        self.cursor.execute("SELECT value FROM value_table WHERE state_key=?", (key_str,))
        row = self.cursor.fetchone()
        return row[0] if row else None

    def _set_db_value(self, state_key, value):
        key_str = self._serialize_key(state_key)
        self.cursor.execute(
            "INSERT OR REPLACE INTO value_table (state_key, value) VALUES (?, ?)", (key_str, value)
        )
        self.conn.commit()

    def _sync_cache_to_db(self):
        for state_key, value in self.value_table.items():
            self._set_db_value(state_key, value)
