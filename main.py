# main.py
# Description: Main execution script for training and playing with an AlphaZero-like chess model.
# It includes supervised and unsupervised training phases, and functionalities to play against the AI.
# The data for training is taken from the Lichess elite database available at:
# https://database.nikonoel.fr/
import sys
print(sys.path)
from src.training import read_games_lazy, process_single_game
from src.model import build_alpha_zero_model, train_model
from src.gameplay import iterative_self_play_and_training, play_against_ai
from tensorflow.keras.models import load_model

def begin_supervised_learning(model, pgn_path):
    """ Conduct supervised learning by processing games from a PGN file. """
    batch_size = 10000
    batch_data = []
    skip = 0
    for game in read_games_lazy(pgn_path):
        if skip < 100:
            skip += 1
            continue
        game_data = process_single_game(game)
        if game_data:
            batch_data.extend(game_data)
            if len(batch_data) >= batch_size:
                train_model(model, batch_data, 1)
                batch_data = []  # Clear batch data after training
    
    if batch_data:  # Train on any remaining data
        train_model(model, batch_data, 1)

def begin_unsupervised_learning(model, cycles, games_per_cycle, iters):
    """ Start unsupervised learning through self-play and training. """
    iterative_self_play_and_training(model, cycles=cycles, games_per_cycle=games_per_cycle, number_of_iterations=iters)

def human_vs_ai(model):
    """ Facilitate playing a game against the trained AI model. """
    play_against_ai(model)

def save_model(model, path):
    """ Save the model to the specified path. """
    model.save(path)

def load_model_from_path(path):
    """ Load a model from the specified path. """
    return load_model(path)

# Example usage
if __name__ == '__main__':

    model = build_alpha_zero_model()
    model.compile(optimizer='adam',
                  loss={'policy_output': 'categorical_crossentropy', 'value_output': 'mean_squared_error'},
                  metrics={'policy_output': 'accuracy', 'value_output': 'mse'})
    # Begin supervised or unsupervised training as required
    begin_supervised_learning(model, 'games_database/lichess_elite_2020-05.pgn')
    # save_model(model, 'path_to_my_model.h5')
    # model = load_model_from_path('path_to_my_model.h5')
    # human_vs_ai(model)
