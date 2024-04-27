import numpy as np
from .board import encodeBoard, encodeMove, decodeMove

def get_historical_representation(current_state, history):
    # Encode the current state
    current_encoded = encodeBoard(current_state)

    # If history is not initialized (empty), create it with the current state
    if history.shape[0] == 0:
        # Initialize history as a 4D array with the first state
        history = np.expand_dims(current_encoded, axis=0)
    else:
        # Append the new state at the end, along the first dimension (axis=0)
        history = np.append(history, np.expand_dims(current_encoded, axis=0), axis=0)
        
        # If the history exceeds 4 entries, remove the oldest (first) entry
        if history.shape[0] > 4:
            history = history[1:]  # Slice off the first entry

    # If there are fewer than 4 states, repeat the first state to fill up the buffer
    while history.shape[0] < 4:
        history = np.insert(history, 0, history[0], axis=0)

    return history

class Node:
    def __init__(self, state, parent=None, history = None):
        self.state = state
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.value = 0.0
        self.untried_actions = list(self.state.legal_moves)
        if isinstance(history, np.ndarray):
            self.history = history
        else:
            # If history is None or not a numpy array, initialize it as an empty numpy array
            # with dimensions (0, 8, 8, 14) which is appropriate for storing state encodings
            self.history = np.empty((0, 8, 8, 14))

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
            new_history = get_historical_representation(next_state, self.history)
            # new_history.append(encodeBoard(next_state))  # Add the new state to the history
            # if len(new_history) > 4:
            #     new_history.pop(0)  # Ensure only the last four states are kept
    
            # Prepare the historical representation for the model
            representation = get_historical_representation(next_state, new_history)
            representation = representation.reshape(1, 4, 8, 8, 14)  # Ensure the correct shape for the model
    
            # Predict using the model
            policy_vector, value_estimate = model.predict(representation, verbose=0)
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


def uct_search(root_state, number_of_iterations, neural_network, history):
    root_node = Node(root_state)

    for _ in range(number_of_iterations):
        node = root_node
        while not node.is_terminal_node():
            if node.is_fully_expanded():
                node = node.best_child()
            else:
                historical_representation = get_historical_representation(root_state, history)
                historical_representation = np.expand_dims(historical_representation, axis=0)  # Add batch dimension
                # print('Historical rep shape: ', historical_representation.shape)
                #print(historical_representation[0,0,:,:,0])
                policy_vector, _ = neural_network.predict(historical_representation, verbose=0)  # Ensure you use .predict() if needed
#                 best_move_index = np.argmax(policy_vector)
#                 best_move = decodeMove(best_move_index, board)  # Implement decodeMove based on your encoding

#                 print("Predicted Best Move:", best_move)
                legal_moves = list(node.state.legal_moves)
                moves_probabilities = np.array([policy_vector[0, encodeMove(move, node.state)] for move in legal_moves])

                # Normalize the probabilities to ensure they sum to 1
                normalized_probabilities = moves_probabilities / np.sum(moves_probabilities)

                # Printing each legal move with its probability
                # print("Legal moves and their probabilities:")
#                 for move, prob in zip(legal_moves, normalized_probabilities):
#                     print(f"Move: {move}, Probability: {prob:.4f}")
                

                # Choose move based on the probabilities
                move = np.random.choice(legal_moves, p=normalized_probabilities)


                if move not in node.untried_actions:
                    node = node.children[move]
                else:
                    node = node.expand(move, neural_network, node.history)
                    break  # Break after expanding to proceed to simulation


    # Simulation and Backpropagation as before...
#     current_state = node.state.copy()
#     while not current_state.is_game_over():
#         legal_moves = list(current_state.legal_moves)

#         # Get historical representation of the current state
#         historical_representation = get_historical_representation(current_state, node.history)
#         historical_representation = historical_representation.reshape(1, 4, 8, 8, 14)  # Assuming this is the shape your network expects

#         # Predict using the neural network
#         policy_vector, value = neural_network.predict(historical_representation, verbose=0)

#         # Normalize the policy vector for only legal moves
#         move_probabilities = np.zeros(policy_vector.shape)
#         legal_indices = [encodeMove(move, current_state) for move in legal_moves]
#         legal_probs = policy_vector[0, legal_indices]
#         legal_probs = legal_probs / np.sum(legal_probs)  # Normalize probabilities

#         # Weighted random choice among legal moves
#         move = np.random.choice(legal_moves, p=legal_probs)

#         # Push the chosen move to the state
#         current_state.push(move)


#     result = current_state.result()
#     numeric_result = 1.0 if result == "1-0" else 0.5 if result == "1/2-1/2" else 0.0
    value_estimate = neural_network.predict(np.expand_dims(get_historical_representation(node.state, node.history), axis=0), verbose=0)[1][0]
    node.backpropagate(value_estimate)

    return root_node.best_child(c_param=0.0).state.move_stack[-1]  # Return the last move leading to the best child