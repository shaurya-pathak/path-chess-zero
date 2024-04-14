# AlphaZero-like Chess Model

This repository contains an implementation of a neural network-based chess engine inspired by DeepMind's AlphaZero. The model combines Monte Carlo Tree Search (MCTS) with deep learning techniques to play chess at a competitive level, learning from games played against itself in a process known as self-play.

## Features

- **Neural Network Model**: Implements a convolutional neural network (CNN) with residual and squeeze-and-excitation (SE) blocks designed to process the spatial and temporal aspects of the chess board state across multiple previous moves.
- **Monte Carlo Tree Search (MCTS)**: Utilizes an enhanced version of MCTS for move decision-making, integrating neural network evaluations for better move exploration and exploitation.
- **Self-Play Training Loop**: Capable of self-training through repeated games where the model learns from its own gameplay, improving over time.
- **Historical State Representation**: Processes the last four board states to capture the temporal dynamics of the game, enhancing the model's strategic depth.

## Installation

1. Clone the repository:
   git clone https://github.com/your-github/alphazero-chess.git
2. Install required packages:
   pip install -r requirements.txt

## Usage

To start a self-play training session:
from self_play import iterative_self_play_and_training

model = build_alpha_zero_model()
iterative_self_play_and_training(model, cycles=1000, games_per_cycle=10, number_of_iterations=200)

## Model Architecture

The model is built using TensorFlow and Keras, with the following layers:
- Input layer that accepts a (4, 8, 8, 14) shape representing the last four states of the chess board.
- Residual blocks for deep feature extraction.
- Squeeze-and-excitation blocks for feature recalibration.
- Separate heads for policy (move probabilities) and value (game outcome predictions) outputs.

## Evaluation

To evaluate the model, run self-play games and observe the improvement over time in the model's decision-making abilities and its success against baseline models or previous versions of itself.

## Contributing

Contributions to this project are welcome! Please fork the repository and submit pull requests with any enhancements, bug fixes, or improvements.

## License

Distributed under the MIT License. See `LICENSE` for more information.
