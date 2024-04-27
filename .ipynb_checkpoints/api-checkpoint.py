from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import chess
from src.board import encodeBoard, get_historical_representation
from src.mcts import uct_search
from main import load_model

app = Flask(__name__)
CORS(app)

def evaluate_best_move(model, fen_strings, number_of_iterations=100):
    if len(fen_strings) != 4:
        raise ValueError("Exactly four FEN strings are required, representing the current board and three historical states.")

    # Initialize the board states from the FEN strings
    boards = [chess.Board(fen) for fen in fen_strings]
    
    # Convert historical board states to the required format and initialize history as a numpy array
    history = np.array([encodeBoard(board) for board in boards[:-1]])  # Encoding the first three boards

    current_board = boards[-1]
    if current_board.is_game_over():
        return "Game over for this board state."

    # Getting the historical representation including the current board
    historical_representation = get_historical_representation(current_board, history)
    
    # Conducting the UCT search to find the best move
    best_move = uct_search(current_board, number_of_iterations, model, historical_representation)

    # Returning the move in Standard Algebraic Notation (SAN) or a message if no move is found
    if best_move:
        return current_board.san(best_move)
    else:
        return "No valid move found."

model = load_model("models/alpha-path_model")

@app.route('/evaluate_move', methods=['POST'])
def evaluate_move():
    data = request.get_json()
    fen_strings = data.get('fen_strings')
    if not fen_strings or len(fen_strings) != 4:
        return jsonify({'error': 'Please provide exactly four FEN strings.'}), 400
    try:
        best_move = evaluate_best_move(model, fen_strings)
        return jsonify({'best_move': best_move}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
