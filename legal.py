"""
Check detection and legal-move filtering.

A pseudo-legal move (from moves.py) becomes truly legal only if it does not
leave the moving side's king in check.  This module adds that filter, plus
the three special moves that require tracking game history: en passant,
castling, and pawn promotion.
"""

import numpy as np
from board import piece_at, apply_move, WHITE, BLACK, PAWN, BISHOP, KNIGHT, ROOK, QUEEN, KING
from moves import pawn_moves, bishop_moves, knight_moves, rook_moves, queen_moves, king_moves


# ---------------------------------------------------------------------------
# Finding the king
# ---------------------------------------------------------------------------

def find_king(M, color):
    """Return (row, col) of `color`'s king."""
    target = color * KING
    for r in range(8):
        for c in range(8):
            if M[r, c] == target:
                return (r, c)
    raise ValueError("No king on board — invalid state")


# ---------------------------------------------------------------------------
# Attack detection
# ---------------------------------------------------------------------------

def is_attacked(M, r, c, by_color):
    """
    Return True if the square (r, c) is attacked by any piece of `by_color`.

    We check each piece type separately using its own attack pattern,
    looking outward from (r, c) as if the square itself were the attacker.
    This reverse-lookup trick means we never have to enumerate every enemy
    piece on the board.
    """
    enemy = by_color

    # --- Knights ---
    for dr, dc in [(1,2),(2,1),(-1,2),(-2,1),(1,-2),(2,-1),(-1,-2),(-2,-1)]:
        rr, cc = r + dr, c + dc
        if 0 <= rr < 8 and 0 <= cc < 8:
            if M[rr, cc] == enemy * KNIGHT:
                return True

    # --- Rooks and queens along ranks / files ---
    for dr, dc in [(1,0),(-1,0),(0,1),(0,-1)]:
        rr, cc = r + dr, c + dc
        while 0 <= rr < 8 and 0 <= cc < 8:
            v = M[rr, cc]
            if v != 0:
                if v == enemy * ROOK or v == enemy * QUEEN:
                    return True
                break          # blocked by any other piece
            rr += dr
            cc += dc

    # --- Bishops and queens along diagonals ---
    for dr, dc in [(1,1),(1,-1),(-1,1),(-1,-1)]:
        rr, cc = r + dr, c + dc
        while 0 <= rr < 8 and 0 <= cc < 8:
            v = M[rr, cc]
            if v != 0:
                if v == enemy * BISHOP or v == enemy * QUEEN:
                    return True
                break
            rr += dr
            cc += dc

    # --- Pawns ---
    # White pawns move toward row 0 (direction -1) and attack diagonally upward.
    # A white pawn at (r+1, c±1) attacks (r, c).
    # A black pawn at (r-1, c±1) attacks (r, c).
    pawn_row = r + 1 if by_color == WHITE else r - 1
    for dc in (-1, 1):
        cc = c + dc
        if 0 <= pawn_row < 8 and 0 <= cc < 8:
            if M[pawn_row, cc] == enemy * PAWN:
                return True

    # --- King ---
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            rr, cc = r + dr, c + dc
            if 0 <= rr < 8 and 0 <= cc < 8:
                if M[rr, cc] == enemy * KING:
                    return True

    return False


def is_in_check(M, color):
    """Return True if `color`'s king is currently under attack."""
    kr, kc = find_king(M, color)
    return is_attacked(M, kr, kc, -color)


# ---------------------------------------------------------------------------
# Legal-move generation
# ---------------------------------------------------------------------------

_PIECE_MOVE_FN = {
    PAWN:   pawn_moves,
    BISHOP: bishop_moves,
    KNIGHT: knight_moves,
    ROOK:   rook_moves,
    QUEEN:  queen_moves,
    KING:   king_moves,
}


def legal_moves(M, r, c, ep_target=None, castling_rights=None):
    """
    Return all truly legal destination squares for the piece at (r, c).

    Parameters
    ----------
    M               : 8x8 board matrix
    r, c            : square of the piece to move
    ep_target       : (row, col) of the en-passant capture square, or None
    castling_rights : dict with boolean values for keys 'K','Q','k','q', or None

    Returns a list of (row, col) destination tuples.
    """
    piece = piece_at(M, r, c)
    if piece == 0:
        return []

    color      = int(np.sign(piece))
    piece_type = abs(piece)

    # Step 1: collect pseudo-legal moves from moves.py
    pseudo = _PIECE_MOVE_FN[piece_type](M, r, c)

    # Step 2: add en-passant as a candidate pawn move
    if piece_type == PAWN and ep_target is not None:
        ep_r, ep_c = ep_target
        direction  = -1 if color == WHITE else 1
        if ep_r == r + direction and abs(ep_c - c) == 1:
            pseudo.append(ep_target)

    # Step 3: filter — keep only moves that don't leave own king in check
    valid = []
    for dst in pseudo:
        M2 = apply_move(M, (r, c), dst)

        # En-passant capture: the taken pawn is on the same rank as the
        # attacker, not on the destination square, so remove it manually.
        if piece_type == PAWN and ep_target is not None and dst == ep_target:
            M2[r, dst[1]] = 0

        if not is_in_check(M2, color):
            valid.append(dst)

    # Step 4: castling (handled separately because it has its own legality rules)
    if piece_type == KING and castling_rights:
        valid += _castling_moves(M, r, c, color, castling_rights)

    return valid


def _castling_moves(M, king_r, king_c, color, castling_rights):
    """
    Return kingside and/or queenside castling destinations that are currently
    legal.  Three conditions must ALL hold for each side:
      1. The castling right has not been forfeited (king and rook never moved).
      2. All squares between king and rook are empty.
      3. The king does not start in check, pass through check, or land in check.
    """
    if is_in_check(M, color):   # can't castle out of check
        return []

    enemy  = -color
    moves  = []

    if color == WHITE:
        # Kingside (king e1→g1, rook h1→f1)
        if (castling_rights.get('K')
                and M[7, 5] == 0 and M[7, 6] == 0
                and not is_attacked(M, 7, 5, enemy)
                and not is_attacked(M, 7, 6, enemy)):
            moves.append((7, 6))
        # Queenside (king e1→c1, rook a1→d1)
        if (castling_rights.get('Q')
                and M[7, 3] == 0 and M[7, 2] == 0 and M[7, 1] == 0
                and not is_attacked(M, 7, 3, enemy)
                and not is_attacked(M, 7, 2, enemy)):
            moves.append((7, 2))
    else:
        # Kingside (king e8→g8, rook h8→f8)
        if (castling_rights.get('k')
                and M[0, 5] == 0 and M[0, 6] == 0
                and not is_attacked(M, 0, 5, enemy)
                and not is_attacked(M, 0, 6, enemy)):
            moves.append((0, 6))
        # Queenside (king e8→c8, rook a8→d8)
        if (castling_rights.get('q')
                and M[0, 3] == 0 and M[0, 2] == 0 and M[0, 1] == 0
                and not is_attacked(M, 0, 3, enemy)
                and not is_attacked(M, 0, 2, enemy)):
            moves.append((0, 2))

    return moves


def all_legal_moves(M, color, ep_target=None, castling_rights=None):
    """
    Return every legal (src, dst) pair for all pieces of `color`.
    Used to detect checkmate and stalemate.
    """
    moves = []
    for r in range(8):
        for c in range(8):
            if int(np.sign(M[r, c])) == color:
                for dst in legal_moves(M, r, c, ep_target, castling_rights):
                    moves.append(((r, c), dst))
    return moves
