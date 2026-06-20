"""
Game state and command-line game loop.

GameState wraps the board matrix with the extra bookkeeping that a complete
chess game requires: whose turn it is, castling rights, en-passant target,
the half-move clock for the 50-move rule, and a position history for the
threefold-repetition rule.

Run this file directly to play a game:
    python game.py
"""

import numpy as np
from board import (
    initial_board, render, apply_move, piece_at,
    WHITE, BLACK, PAWN, BISHOP, KNIGHT, ROOK, QUEEN, KING,
)
from legal import legal_moves, is_in_check, all_legal_moves


# ---------------------------------------------------------------------------
# GameState
# ---------------------------------------------------------------------------

class GameState:
    """All information needed to fully describe a chess position."""

    def __init__(self):
        self.M     = initial_board()
        self.turn  = WHITE          # white always moves first

        # Each key becomes False the moment the relevant piece moves.
        # 'K'/'Q' = white kingside/queenside; 'k'/'q' = black.
        self.castling_rights = {'K': True, 'Q': True, 'k': True, 'q': True}

        # The square a pawn can capture into via en passant, or None.
        self.ep_target = None

        # Counts half-moves since the last pawn move or capture.
        # The game is drawn when this reaches 100 (the 50-move rule).
        self.halfmove_clock = 0

        # Incremented after every Black move.
        self.fullmove = 1

        # History of board states (as hashable tuples) for threefold repetition.
        self.position_history = []
        self._record_position()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _record_position(self):
        """Snapshot the board + key state for repetition detection."""
        key = (
            self.M.tobytes(),
            self.turn,
            tuple(sorted(k for k, v in self.castling_rights.items() if v)),
            self.ep_target,
        )
        self.position_history.append(key)

    def _current_legal_moves(self):
        return all_legal_moves(self.M, self.turn, self.ep_target, self.castling_rights)

    # ------------------------------------------------------------------
    # Game-over detection
    # ------------------------------------------------------------------

    def is_game_over(self):
        """
        Return (True, message) if the game has ended, else (False, None).

        Checks (in order): checkmate, stalemate, 50-move rule,
        threefold repetition, and insufficient material.
        """
        moves = self._current_legal_moves()

        if not moves:
            if is_in_check(self.M, self.turn):
                winner = "Black" if self.turn == WHITE else "White"
                return True, f"Checkmate — {winner} wins."
            return True, "Stalemate — draw."

        if self.halfmove_clock >= 100:
            return True, "Draw by the 50-move rule."

        if self.position_history.count(self.position_history[-1]) >= 3:
            return True, "Draw by threefold repetition."

        if self._insufficient_material():
            return True, "Draw by insufficient material."

        return False, None

    def _insufficient_material(self):
        """True when neither side has enough pieces to deliver checkmate."""
        pieces = [abs(v) for row in self.M for v in row if v != 0]
        pieces = [p for p in pieces if p != KING]
        if not pieces:
            return True
        if pieces == [KNIGHT] or pieces == [BISHOP]:
            return True
        return False

    # ------------------------------------------------------------------
    # Applying a move
    # ------------------------------------------------------------------

    def make_move(self, src, dst, promotion=QUEEN):
        """
        Execute a move, updating the board and all auxiliary state.

        `promotion` is the piece type (QUEEN/ROOK/BISHOP/KNIGHT) that a
        pawn becomes when it reaches the back rank.  Defaults to queen.
        """
        r,  c  = src
        r2, c2 = dst
        piece      = piece_at(self.M, r, c)
        piece_type = abs(piece)
        captured   = piece_at(self.M, r2, c2)

        # --- 1. Move the piece ---
        self.M = apply_move(self.M, src, dst)

        # --- 2. En passant: the captured pawn is NOT on the destination ---
        if piece_type == PAWN and self.ep_target == (r2, c2):
            self.M[r, c2] = 0      # remove the pawn that made the double push

        # --- 3. Castling: slide the rook to its post-castle square ---
        if piece_type == KING:
            rook_row = r           # king and rooks share the same rank
            if c2 - c == 2:       # kingside: king moved right two squares
                self.M = apply_move(self.M, (rook_row, 7), (rook_row, 5))
            elif c - c2 == 2:     # queenside: king moved left two squares
                self.M = apply_move(self.M, (rook_row, 0), (rook_row, 3))

        # --- 4. Pawn promotion ---
        if piece_type == PAWN and (r2 == 0 or r2 == 7):
            self.M[r2, c2] = self.turn * promotion

        # --- 5. Update castling rights ---
        # King move forfeits both rights for that colour.
        if piece_type == KING:
            if self.turn == WHITE:
                self.castling_rights['K'] = self.castling_rights['Q'] = False
            else:
                self.castling_rights['k'] = self.castling_rights['q'] = False

        # Rook move forfeits the right on its side.
        if piece_type == ROOK:
            _rook_right_lost = {(7,7):'K', (7,0):'Q', (0,7):'k', (0,0):'q'}
            right = _rook_right_lost.get((r, c))
            if right:
                self.castling_rights[right] = False

        # A rook captured on its home square also forfeits that right.
        if captured != 0:
            _rook_right_lost = {(7,7):'K', (7,0):'Q', (0,7):'k', (0,0):'q'}
            right = _rook_right_lost.get((r2, c2))
            if right:
                self.castling_rights[right] = False

        # --- 6. Set the new en-passant target ---
        if piece_type == PAWN and abs(r2 - r) == 2:
            self.ep_target = ((r + r2) // 2, c)   # the square the pawn skipped over
        else:
            self.ep_target = None

        # --- 7. Half-move clock (resets on pawn moves and captures) ---
        if piece_type == PAWN or captured != 0:
            self.halfmove_clock = 0
        else:
            self.halfmove_clock += 1

        # --- 8. Full-move counter (increments after Black's move) ---
        if self.turn == BLACK:
            self.fullmove += 1

        # --- 9. Switch sides ---
        self.turn = -self.turn

        # --- 10. Record position for repetition detection ---
        self._record_position()


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

FILES = 'abcdefgh'

def parse_move(text):
    """
    Convert a move string like 'e2e4' or 'e7e8q' into
    ((src_row, src_col), (dst_row, dst_col), promotion_piece_type).

    Returns (None, None, None) if the string is not valid.
    """
    text = text.strip().lower()
    if len(text) < 4:
        return None, None, None
    try:
        c1 = FILES.index(text[0])
        r1 = 8 - int(text[1])
        c2 = FILES.index(text[2])
        r2 = 8 - int(text[3])
    except (ValueError, IndexError):
        return None, None, None

    promo_map = {'q': QUEEN, 'r': ROOK, 'b': BISHOP, 'n': KNIGHT}
    promotion = promo_map.get(text[4] if len(text) >= 5 else '', QUEEN)

    return (r1, c1), (r2, c2), promotion


def square_name(r, c):
    """Convert (row, col) to algebraic notation, e.g. (6, 4) → 'e2'."""
    return FILES[c] + str(8 - r)


# ---------------------------------------------------------------------------
# Game loop
# ---------------------------------------------------------------------------

def play():
    """Run an interactive two-player chess game in the terminal."""
    gs = GameState()

    while True:
        print('\n' + render(gs.M))

        over, result = gs.is_game_over()
        if over:
            print('\n' + result)
            break

        turn_name  = "White" if gs.turn == WHITE else "Black"
        check_note = "  *** CHECK ***" if is_in_check(gs.M, gs.turn) else ""
        print(f"\nMove {gs.fullmove}  |  {turn_name} to play{check_note}")
        print("Enter a move (e.g. e2e4, e7e8q) or 'quit': ", end='', flush=True)

        raw = input()
        if raw.strip().lower() == 'quit':
            print("Game abandoned.")
            break

        src, dst, promotion = parse_move(raw)
        if src is None:
            print("  Could not understand that.  Use the format: e2e4")
            continue

        allowed = legal_moves(gs.M, src[0], src[1], gs.ep_target, gs.castling_rights)
        if dst not in allowed:
            piece = piece_at(gs.M, src[0], src[1])
            if piece == 0:
                print(f"  There is no piece on {square_name(*src)}.")
            elif int(np.sign(piece)) != gs.turn:
                print(f"  That piece belongs to the other side.")
            else:
                print(f"  Illegal move.  Try again.")
            continue

        gs.make_move(src, dst, promotion)


if __name__ == '__main__':
    play()
