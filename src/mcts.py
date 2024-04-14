import numpy as np

def get_historical_representation(current_state, history):
    # Encode the current state and add to history
    current_encoded = encodeBoard(current_state)
    history.append(current_encoded)

    # Ensure history is no longer than 4 states
    if len(history) > 4:
        history.pop(0)

    # If fewer than 4 states, repeat the first state to fill the buffer
    while len(history) < 4:
        history.append(history[0])

    # Stack the representations along a new axis
    return np.stack(history, axis=0)

class Node:
    def __init__(self, state, parent=None, history = None):
        self.state = state
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.value = 0.0
        self.untried_actions = list(self.state.legal_moves)
        self.history = history if history is not None else []

        self.policy = None  # Placeholder for storing the policy vector for this node, if needed.

    def is_terminal_node(self):
        return self.state.is_game_over()

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def expand(self, action, model, history):
        if action in self.untried_actions:
            self.untried_actions.remove(action)
            next_state = self.state.copy()
            next_state.push(action)
    
            # Use the history passed in to generate the new historical representation
            # This ensures history is passed correctly and updated within each node
            new_history = history.copy()  # Copy the current history
            new_history.append(encodeBoard(next_state))  # Add the new state to the history
            if len(new_history) > 4:
                new_history.pop(0)  # Ensure only the last four states are kept
    
            # Prepare the historical representation for the model
            representation = get_historical_representation(next_state, new_history)
            representation = representation.reshape(1, 4, 8, 8, 14)  # Ensure the correct shape for the model
    
            # Predict using the model
            policy_vector, value_estimate = model.predict(representation)
            child_node = Node(next_state, parent=self, history=new_history)  # Pass the updated history to the new node
            child_node.value = value_estimate
    
            encoded_action = encodeMove(action, next_state)
            child_node.policy = policy_vector[0, encoded_action]
            self.children[action] = child_node
            return child_node
        else:
            raise ValueError(f"Action {action} not available for expansion. Current untried actions: {self.untried_actions}")


    def update(self, value):
        self.visits += 1
        self.value += value  # This might require adjustment depending on whether you're using total value or average value.

    def backpropagate(self, result):
        # Recursively update nodes from the current node back to the root.
        self.update(result)
        if self.parent is not None:
            self.parent.backpropagate(result)


    def best_child(self, c_param=1.4):
        choices_weights = [
            (child.value / child.visits) + c_param * np.sqrt((2 * np.log(self.visits) / child.visits))
            for child in self.children.values()
        ]
        return list(self.children.values())[np.argmax(choices_weights)]

def neural_network_predict(current_eval_state):
    # Assuming 'current_eval_state' is a chess.Board object
    # First, encode the current board state for the model

    encoded_state = encodeBoard(current_eval_state)  # Adjust this function as necessary

    # Reshape the input to match the model's expected input shape (batch size of 1)
    encoded_state = np.expand_dims(encoded_state, axis=0)
    # print("Encoded_state", encoded_state.shape)
    # Make a prediction with the model
    predictions = model.predict(encoded_state)

    # Extract the policy vector and value estimate from the predictions
    policy_vector = predictions[0][0]  # Assuming the first output is the policy vector
    value_estimate = predictions[1][0]  # Assuming the second output is the value estimate

    return policy_vector, value_estimate


def prepare_historical_representation(history):
    # Check if history length is less than required (e.g., 4 states), and pad if necessary
    while len(history) < 4:
        # Duplicate the first element if history is not empty, else use a zero array
        if history:
            history.insert(0, history[0])
        else:
            # Assuming your board encoding is a numpy array with the shape matching your network input
            history.insert(0, np.zeros((8, 8, 14)))  # You need to replace (8, 8, 14) with the actual dimensions of your encoded state

    # If there are more than 4 states, trim the oldest
    if len(history) > 4:
        history = history[-4:]

    # Stack along a new dimension to create a single input tensor
    return np.stack(history, axis=0)


def uct_search(root_state, number_of_iterations, neural_network, history):
    root_node = Node(root_state, history = [])

    for _ in range(number_of_iterations):
        node = root_node
        while not node.is_terminal_node():
            if node.is_fully_expanded():
                node = node.best_child()
            else:
                historical_representation = prepare_historical_representation(node.history)
                historical_representation = np.expand_dims(historical_representation, axis=0)  # Add batch dimension
                policy_vector, _ = neural_network(historical_representation)
                legal_moves = list(node.state.legal_moves)
                moves_probabilities = np.array([policy_vector[0, encodeMove(move, node.state)] for move in legal_moves])
                moves_probabilities /= np.sum(moves_probabilities)
                move = np.random.choice(legal_moves, p=moves_probabilities)

                if move not in node.untried_actions:
                    node = node.children[move]
                else:
                    node = node.expand(move, neural_network, node.history)
                    break  # Break after expanding to proceed to simulation

        # Simulation and Backpropagation as before...
        current_state = node.state.copy()
        while not current_state.is_game_over():
            legal_moves = list(current_state.legal_moves)
            move = np.random.choice(legal_moves)
            current_state.push(move)

        result = current_state.result()
        numeric_result = 1.0 if result == "1-0" else 0.25 if result == "1/2-1/2" else 0.0
        node.backpropagate(numeric_result)

    return root_node.best_child(c_param=0.0).state.move_stack[-1]  # Return the last move leading to the best child
