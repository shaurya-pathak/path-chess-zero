import chess.pgn
import numpy as np
from .board import encodeBoard, encodeMove, get_historical_representation


def read_games_lazy(pgn_file_path):
    games_read = 0
    with open(pgn_file_path) as pgn_file:
        while True:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break
            yield game
            games_read += 1

def process_single_game(game):
    if game is None:
        return None

    game_result = game.headers["Result"]
    white_win = 1 if game_result == "1-0" else (-1 if game_result == "0-1" else 0)

    game_data = []
    board = game.board()
    history = np.empty((0, 8, 8, 14))  # Assuming encodeBoard returns an 8x8x14 board state

    for move in game.mainline_moves():
        historical_representation = get_historical_representation(board, history)
        policy_vector = np.zeros((4672,))
        value_estimate = white_win if board.turn == chess.WHITE else -white_win
        
        move_index = encodeMove(move, board)
        policy_vector[move_index] = 1
        
        board.push(move)
        game_data.append((historical_representation, policy_vector, value_estimate))

        # Update history with the new state directly in the main loop
        current_encoded = encodeBoard(board)
        history = np.append(history, [current_encoded], axis=0)
        if history.shape[0] > 4:
            history = history[1:]

    return game_data
