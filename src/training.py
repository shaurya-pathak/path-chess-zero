import chess.pgn
import numpy as np
from .board import encodeBoard, encodeMove


def read_games_lazy(pgn_file_path):
    games_read = 0
    with open(pgn_file_path) as pgn_file:
        while True:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break
            yield game
            games_read += 1

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

def process_single_game(game):
    if game is None:
        return None

    game_result = game.headers["Result"]
    if game_result == "1-0":
        white_win, black_win = 1, -1
    elif game_result == "0-1":
        white_win, black_win = -1, 1
    else:
        white_win, black_win = 0, 0
    
    game_data = []
    board = game.board()
    history = []  # Initialize the history list

    for move in game.mainline_moves():
        historical_representation = get_historical_representation(board, history)

        policy_vector = np.zeros((4672,))
        value_estimate = white_win if board.turn == chess.WHITE else black_win
    
        move_index = encodeMove(move, board)
        policy_vector[move_index] = 1
        
        board.push(move)
        game_data.append((historical_representation, policy_vector, value_estimate))

    # Invert the value estimate for each state except the last to adjust the perspective
    for i in range(len(game_data) - 1):
        game_data[i] = game_data[i][0], game_data[i][1], -game_data[i][2]

    return game_data
