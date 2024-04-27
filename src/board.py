from pathlib import Path
from typing import Optional
from gym_chess.alphazero.move_encoding import utils
import chess
import numpy as np

def create_move_mapping():
    letters = 'abcdefgh'
    numbers = '12345678'
    promotions = 'qrbn'  # queen, rook, bishop, knight
    move_mapping = {}
    idx = 0

    # Normal moves and captures
    for start_letter in letters:
        for start_number in numbers:
            for end_letter in letters:
                for end_number in numbers:
                    for promotion in ['', *promotions]:
                        move = f'{start_letter}{start_number}{end_letter}{end_number}{promotion}'
                        move_mapping[move] = idx
                        idx += 1

    # Castling moves
    for castling in ['e1g1', 'e1c1', 'e8g8', 'e8c8']:  # king's side and queen's side for both colors
        move_mapping[castling] = idx
        idx += 1

    return move_mapping

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

def filter_legal_moves(board, policy_vector, move_mapping):
    legal_moves = list(board.legal_moves)
    legal_indices = []
    legal_probs = []

    print("Total policy vector size:", len(policy_vector))  # Debug

    for move in legal_moves:
        move_uci = move.uci()
        idx = move_mapping.get(move_uci, None)
        if idx is not None and idx < len(policy_vector):
            legal_indices.append(idx)
            legal_probs.append(policy_vector[idx])
        else:
            print(f"Illegal index or move not found: {move_uci}, idx: {idx}")  # Debug

    # Re-normalize probabilities
    if legal_probs:
        legal_probs = np.array(legal_probs)
        legal_probs /= np.sum(legal_probs)
    else:
        print("No legal moves found in policy vector.")  # Debug

    return legal_moves, legal_probs

def select_move(legal_moves, legal_probs):
    if legal_probs.size > 0:
        move_idx = np.random.choice(range(len(legal_moves)), p=legal_probs)
        selected_move = legal_moves[move_idx]
        print(f"Selected move: {selected_move.uci()} with prob: {legal_probs[move_idx]}")  # Debug
    else:
        print("Defaulting to random move due to empty probabilities.")  # Debug
        selected_move = np.random.choice(legal_moves)

    return selected_move

def _decodeKnight(action: int) -> Optional[chess.Move]:
    _NUM_TYPES: int = 8

    #: Starting point of knight moves in last dimension of 8 x 8 x 73 action array.
    _TYPE_OFFSET: int = 56

    #: Set of possible directions for a knight move, encoded as
    #: (delta rank, delta square).
    _DIRECTIONS = utils.IndexedTuple(
        (+2, +1),
        (+1, +2),
        (-1, +2),
        (-2, +1),
        (-2, -1),
        (-1, -2),
        (+1, -2),
        (+2, -1),
    )

    from_rank, from_file, move_type = np.unravel_index(action, (8, 8, 73))

    is_knight_move = (
        _TYPE_OFFSET <= move_type
        and move_type < _TYPE_OFFSET + _NUM_TYPES
    )

    if not is_knight_move:
        return None

    knight_move_type = move_type - _TYPE_OFFSET

    delta_rank, delta_file = _DIRECTIONS[knight_move_type]

    to_rank = from_rank + delta_rank
    to_file = from_file + delta_file

    move = utils.pack(from_rank, from_file, to_rank, to_file)
    return move

def _decodeQueen(action: int) -> Optional[chess.Move]:

    _NUM_TYPES: int = 56 # = 8 directions * 7 squares max. distance

    #: Set of possible directions for a queen move, encoded as
    #: (delta rank, delta square).
    _DIRECTIONS = utils.IndexedTuple(
        (+1,  0),
        (+1, +1),
        ( 0, +1),
        (-1, +1),
        (-1,  0),
        (-1, -1),
        ( 0, -1),
        (+1, -1),
    )
    from_rank, from_file, move_type = np.unravel_index(action, (8, 8, 73))

    is_queen_move = move_type < _NUM_TYPES

    if not is_queen_move:
        return None

    direction_idx, distance_idx = np.unravel_index(
        indices=move_type,
        shape=(8,7)
    )

    direction = _DIRECTIONS[direction_idx]
    distance = distance_idx + 1

    delta_rank = direction[0] * distance
    delta_file = direction[1] * distance

    to_rank = from_rank + delta_rank
    to_file = from_file + delta_file

    move = utils.pack(from_rank, from_file, to_rank, to_file)
    return move

def _decodeUnderPromotion(action):
    _NUM_TYPES: int = 9 # = 3 directions * 3 piece types (see below)

    #: Starting point of underpromotions in last dimension of 8 x 8 x 73 action
    #: array.
    _TYPE_OFFSET: int = 64

    #: Set of possibel directions for an underpromotion, encoded as file delta.
    _DIRECTIONS = utils.IndexedTuple(
        -1,
        0,
        +1,
    )

    #: Set of possibel piece types for an underpromotion (promoting to a queen
    #: is implicitly encoded by the corresponding queen move).
    _PROMOTIONS = utils.IndexedTuple(
        chess.KNIGHT,
        chess.BISHOP,
        chess.ROOK,
    )

    from_rank, from_file, move_type = np.unravel_index(action, (8, 8, 73))

    is_underpromotion = (
        _TYPE_OFFSET <= move_type
        and move_type < _TYPE_OFFSET + _NUM_TYPES
    )

    if not is_underpromotion:
        return None

    underpromotion_type = move_type - _TYPE_OFFSET

    direction_idx, promotion_idx = np.unravel_index(
        indices=underpromotion_type,
        shape=(3,3)
    )

    direction = _DIRECTIONS[direction_idx]
    promotion = _PROMOTIONS[promotion_idx]

    to_rank = from_rank + 1
    to_file = from_file + direction

    move = utils.pack(from_rank, from_file, to_rank, to_file)
    move.promotion = promotion

    return move

#primary decoding function, the ones above are just helper functions
def decodeMove(action: int, board) -> chess.Move:
        move = _decodeQueen(action)
        is_queen_move = move is not None

        if not move:
            move = _decodeKnight(action)

        if not move:
            move = _decodeUnderPromotion(action)

        if not move:
            raise ValueError(f"{action} is not a valid action")

        # Actions encode moves from the perspective of the current player. If
        # this is the black player, the move must be reoriented.
        turn = board.turn

        if turn == False: #black to move
            move = utils.rotate(move)

        # Moving a pawn to the opponent's home rank with a queen move
        # is automatically assumed to be queen underpromotion. However,
        # since queenmoves has no reference to the board and can thus not
        # determine whether the moved piece is a pawn, you have to add this
        # information manually here
        if is_queen_move:
            to_rank = chess.square_rank(move.to_square)
            is_promoting_move = (
                (to_rank == 7 and turn == True) or
                (to_rank == 0 and turn == False)
            )

            piece = board.piece_at(move.from_square)
            if piece is None: #NOTE I added this, not entirely sure if it's correct
                return None
            is_pawn = piece.piece_type == chess.PAWN

            if is_pawn and is_promoting_move:
                move.promotion = chess.QUEEN

        return move

def encodeBoard(board: chess.Board) -> np.array:
     """Converts a board to numpy array representation."""

     array = np.zeros((8, 8, 14), dtype=int)

     for square, piece in board.piece_map().items():
      rank, file = chess.square_rank(square), chess.square_file(square)
      piece_type, color = piece.piece_type, piece.color

      # The first six planes encode the pieces of the active player,
      # the following six those of the active player's opponent. Since
      # this class always stores boards oriented towards the white player,
      # White is considered to be the active player here.
      offset = 0 if color == chess.WHITE else 6

      # Chess enumerates piece types beginning with one, which you have
      # to account for
      idx = piece_type - 1

      array[rank, file, idx + offset] = 1

     # Repetition counters
     array[:, :, 12] = board.is_repetition(2)
     array[:, :, 13] = board.is_repetition(3)


     return array

def encodeKnight(move: chess.Move):
    _NUM_TYPES: int = 8

    #: Starting point of knight moves in last dimension of 8 x 8 x 73 action array.
    _TYPE_OFFSET: int = 56

    #: Set of possible directions for a knight move, encoded as
    #: (delta rank, delta square).
    _DIRECTIONS = utils.IndexedTuple(
        (+2, +1),
        (+1, +2),
        (-1, +2),
        (-2, +1),
        (-2, -1),
        (-1, -2),
        (+1, -2),
        (+2, -1),
    )

    from_rank, from_file, to_rank, to_file = utils.unpack(move)
    delta = (to_rank - from_rank, to_file - from_file)
    is_knight_move = delta in _DIRECTIONS

    if not is_knight_move:
        return None

    knight_move_type = _DIRECTIONS.index(delta)
    move_type = _TYPE_OFFSET + knight_move_type

    action = np.ravel_multi_index(
        multi_index=((from_rank, from_file, move_type)),
        dims=(8, 8, 73)
    )

    return action

def encodeQueen(move: chess.Move):
    _NUM_TYPES: int = 56 # = 8 directions * 7 squares max. distance
    _DIRECTIONS = utils.IndexedTuple(
        (+1,  0),
        (+1, +1),
        ( 0, +1),
        (-1, +1),
        (-1,  0),
        (-1, -1),
        ( 0, -1),
        (+1, -1),
    )

    from_rank, from_file, to_rank, to_file = utils.unpack(move)

    delta = (to_rank - from_rank, to_file - from_file)

    is_horizontal = delta[0] == 0
    is_vertical = delta[1] == 0
    is_diagonal = abs(delta[0]) == abs(delta[1])
    is_queen_move_promotion = move.promotion in (chess.QUEEN, None)

    is_queen_move = (
        (is_horizontal or is_vertical or is_diagonal)
            and is_queen_move_promotion
    )

    if not is_queen_move:
        return None

    direction = tuple(np.sign(delta))
    distance = np.max(np.abs(delta))

    direction_idx = _DIRECTIONS.index(direction)
    distance_idx = distance - 1

    move_type = np.ravel_multi_index(
        multi_index=([direction_idx, distance_idx]),
        dims=(8,7)
    )

    action = np.ravel_multi_index(
        multi_index=((from_rank, from_file, move_type)),
        dims=(8, 8, 73)
    )

    return action

def encodeUnder(move):
    _NUM_TYPES: int = 9 # = 3 directions * 3 piece types (see below)
    _TYPE_OFFSET: int = 64
    _DIRECTIONS = utils.IndexedTuple(
        -1,
        0,
        +1,
    )
    _PROMOTIONS = utils.IndexedTuple(
        chess.KNIGHT,
        chess.BISHOP,
        chess.ROOK,
    )

    from_rank, from_file, to_rank, to_file = utils.unpack(move)
    #print(from_rank, from_file, to_rank, to_file)

    is_underpromotion = (
        move.promotion in _PROMOTIONS
        and ((from_rank == 6
        and to_rank == 7)  or from_rank == 1 and to_rank == 0)
    )

    if not is_underpromotion:
        return None

    delta_file = to_file - from_file

    direction_idx = _DIRECTIONS.index(delta_file)
    promotion_idx = _PROMOTIONS.index(move.promotion)

    underpromotion_type = np.ravel_multi_index(
        multi_index=([direction_idx, promotion_idx]),
        dims=(3,3)
    )

    move_type = _TYPE_OFFSET + underpromotion_type

    action = np.ravel_multi_index(
        multi_index=((from_rank, from_file, move_type)),
        dims=(8, 8, 73)
    )

    return action

def encodeMove(move: str, board) -> int:
    # print("Move to be encoded: ", move)
    # move = chess.Move.from_uci(move)
    if board.turn == chess.BLACK:
        move = utils.rotate(move)

    action = encodeQueen(move)

    if action is None:
        action = encodeKnight(move)

    if action is None:
        action = encodeUnder(move)

    if action is None:
        raise ValueError(f"{move} is not a valid move")

    return action

