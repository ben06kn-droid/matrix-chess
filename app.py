"""
Flask web server for Matrix Chess.

Exposes four endpoints:
  GET  /                        -- serve the board UI
  GET  /state                   -- current board state as JSON
  GET  /legal_moves/<r>/<c>     -- legal destinations for the piece at (r, c)
  POST /move   {src, dst, promotion}  -- apply a move
  POST /reset                   -- start a new game
"""

import numpy as np
from flask import Flask, jsonify, request, render_template
from board import WHITE, BLACK, PAWN, BISHOP, KNIGHT, ROOK, QUEEN, KING
from legal import legal_moves, is_in_check
from game import GameState

app = Flask(__name__)
gs = GameState()

SYMBOLS = {
    (WHITE, PAWN): '♙', (WHITE, BISHOP): '♗', (WHITE, KNIGHT): '♘',
    (WHITE, ROOK): '♖', (WHITE, QUEEN):  '♕', (WHITE, KING):   '♔',
    (BLACK, PAWN): '♟', (BLACK, BISHOP): '♝', (BLACK, KNIGHT): '♞',
    (BLACK, ROOK): '♜', (BLACK, QUEEN):  '♛', (BLACK, KING):   '♚',
}


def serialize(state):
    """Convert a GameState to a JSON-friendly dict."""
    board = []
    for r in range(8):
        row = []
        for c in range(8):
            v = state.M[r, c]
            if v == 0:
                row.append(None)
            else:
                color = int(np.sign(v))
                ptype = abs(int(v))
                row.append({
                    'symbol': SYMBOLS[(color, ptype)],
                    'color': 'white' if color == WHITE else 'black',
                    'type': ptype,
                })
        board.append(row)

    over, msg = state.is_game_over()
    in_check = is_in_check(state.M, state.turn) if not over else False

    return {
        'board': board,
        'turn': 'white' if state.turn == WHITE else 'black',
        'in_check': in_check,
        'game_over': over,
        'message': msg or '',
        'ep_target': list(state.ep_target) if state.ep_target else None,
    }


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/state')
def state():
    return jsonify(serialize(gs))


@app.route('/legal_moves/<int:r>/<int:c>')
def get_legal_moves(r, c):
    moves = legal_moves(gs.M, r, c, gs.ep_target, gs.castling_rights)
    return jsonify({'moves': [list(m) for m in moves]})


@app.route('/move', methods=['POST'])
def make_move():
    global gs
    data = request.json
    src = tuple(data['src'])
    dst = tuple(data['dst'])
    promotion = int(data.get('promotion', QUEEN))

    allowed = legal_moves(gs.M, src[0], src[1], gs.ep_target, gs.castling_rights)
    if dst not in allowed:
        return jsonify({'error': 'Illegal move'}), 400

    gs.make_move(src, dst, promotion)
    return jsonify(serialize(gs))


@app.route('/reset', methods=['POST'])
def reset():
    global gs
    gs = GameState()
    return jsonify(serialize(gs))


if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
