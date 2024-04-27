import chess
import datetime
import numpy as np
from IPython.display import HTML, display
from IPython.display import display, clear_output
from ipywidgets import widgets
import chess.svg
import chess.pgn
import datetime
from io import StringIO
from .board import encodeBoard, get_historical_representation
from .mcts import uct_search
import os

def self_play_game(model, number_of_iterations=50, game_index=1):
    board = chess.Board()
    game_data = []
    game = chess.pgn.Game()
    game.headers["Event"] = "Self-play training session"
    game.headers["Site"] = "Local"
    game.headers["Date"] = datetime.datetime.now().strftime("%Y.%m.%d")
    game.headers["Round"] = str(game_index)
    game.headers["White"] = "Model"
    game.headers["Black"] = "Model"
    node = game

    # Initialize history as an empty numpy array with correct dimensions
    history = np.empty((0, 8, 8, 14))  # Assuming encodeBoard returns (8, 8, 14) encoded board state

    while not board.is_game_over():
        board_svg = chess.svg.board(board=board, size=200)
        display(HTML(board_svg))

        # Use historical representation for UCT search
        historical_representation = get_historical_representation(board, history)
        best_move = uct_search(board, number_of_iterations, model, history)

        if best_move:
            board.push(best_move)
            node = node.add_variation(best_move)

            # Update history with the new state
            current_encoded = encodeBoard(board)
            history = np.append(history, [current_encoded], axis=0)
            if history.shape[0] > 4:
                history = history[1:]  # Keep the last 4 states

            # Now use the updated history to get the representation for prediction
            historical_representation = get_historical_representation(board, history)
            predictions = model.predict(np.expand_dims(historical_representation, axis=0))
            policy_vector = predictions[0][0]
            value_estimate = predictions[1][0] if len(predictions) > 1 else None
            game_data.append((historical_representation, policy_vector, value_estimate))
        else:
            print("No valid move found, skipping turn.")

    game.headers["Result"] = board.result()
    return game_data, game


def save_game_to_pgn(game, filename="games.pgn"):
    with open(filename, "a") as pgn_file:
        print(game, file=pgn_file, end="\n\n")

def iterative_self_play_and_training(model, cycles=10, games_per_cycle=10, number_of_iterations=50):
    for cycle in range(cycles):
        all_game_data = []
        for game_index in range(games_per_cycle):
            print('NEW GAME STARTED')
            game_data, game = self_play_game(model, number_of_iterations, game_index + 1)
            all_game_data.extend(game_data)
            print(game)
            print('saving game')
            directory = f"chess_games/games_cycle_{cycle+1}"
            if not os.path.exists(directory):
                os.makedirs(directory)
            save_game_to_pgn(game, filename=os.path.join(directory, f"{game_index+1}.pgn"))
        train_model(model, all_game_data)
        # Additional steps like model evaluation and saving checkpoints can be added here

    print("Training and self-play cycle completed.")



def play_against_ai(model, number_of_iterations=50):
    board = chess.Board()
    game = chess.pgn.Game()
    game.headers["Event"] = "Human vs AI"
    game.headers["Site"] = "Local"
    game.headers["Date"] = datetime.datetime.now().strftime("%Y.%m.%d")
    game.headers["Round"] = "1"
    game.headers["White"] = "Human"
    game.headers["Black"] = "AI"
    node = game

    move_text = widgets.Text(
        description='Your move:',
        disabled=False
    )

    def on_move_submit(sender):
      move = sender.value  # Get the new move
      print(move)
      try:
          chess_move = chess.Move.from_uci(move)
          if chess_move in board.legal_moves:
              board.push(chess_move)
              node.add_variation(chess_move)
              sender.value = ''  # Clear the input field
    
              # AI's turn
              if not board.is_game_over():
                  print("AI is thinking...")
                  best_move = uct_search(board, number_of_iterations, model)
                  if best_move:
                      board.push(best_move)
                      node.add_variation(best_move)
                      print(f"AI move: {best_move.uci()}")
                  else:
                      print("AI failed to find a valid move.")
                  clear_and_display_board()
              else:
                  end_game()
          else:
              print("Illegal move. Try again.")
      except ValueError:
          print("Invalid move format. Use UCI notation (e.g., e2e4). Try again.")
    
    move_text = widgets.Text(description='Your move:')
    move_text.on_submit(on_move_submit)
    display(move_text)

def clear_and_display_board():
    clear_output(wait=True)
    display(HTML(chess.svg.board(board=board, size = 200)))
    display(move_text)

def end_game():
  clear_output(wait=True)
  display(chess.svg.board(board=board, size=400))
  game.headers["Result"] = board.result()
  print(f"Game over. Result: {board.result()}")

  # Save the PGN to a string
  exporter = chess.pgn.StringExporter(headers=True, variations=True, comments=True)
  pgn_string = game.accept(exporter)

  # Optionally, print the PGN string to the output
  print(pgn_string)

  # Save the PGN to a file
  pgn_filename = f"game_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pgn"
  with open(pgn_filename, 'w') as pgn_file:
      pgn_file.write(pgn_string)

  print(f"PGN saved to {pgn_filename}")
