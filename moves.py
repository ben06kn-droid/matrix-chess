"""
Move generation built on top of the linear-algebra primitives in board.py.

Knight and king moves are pure shift-and-mask operations.
Sliding pieces (rook/bishop/queen) find their first obstruction with a
triangular prefix-sum matrix multiplication, not a hand-rolled walk.
Pawns are where genuine if/else lives -- direction, double-step, and
diagonal-only captures are inherently asymmetric chess rules, not
linear-algebra facts -- but even the pawn's forward square is found
with shift(), not coordinate arithmetic.
"""

import numpy as np
from board import shift, piece_at, WHITE

KNIGHT_OFFSETS = [(1, 2), (2, 1), (-1, 2), (-2, 1), (1, -2), (2, -1), (-1, -2), (-2, -1)]
KING_OFFSETS = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]
ROOK_DIRS = [(1, 0), (-1, 0), (0, 1), (0, -1)]
BISHOP_DIRS = [(1, 1), (1, -1), (-1, 1), (-1, -1)]

_LOWER = np.tril(np.ones((8, 8), dtype=int))  # prefix-sum operator


def _offset_squares(M, r, c, offsets, color):
    """Shared engine for knight & king: shift the source square by each
    offset using matrix multiplication, keep landings that are on the
    board and not occupied by a friendly piece."""
    dests = []
    src = np.zeros((8, 8), dtype=int)
    src[r, c] = 1
    for dr, dc in offsets:
        candidate = shift(src, dr, dc)            # pure linear algebra
        if candidate.sum() == 0:
            continue                               # fell off the board
        rr, cc = (int(x) for x in np.argwhere(candidate)[0])
        occupant = piece_at(M, rr, cc)
        if occupant == 0 or np.sign(occupant) != color:   # the if/else layer
            dests.append((rr, cc))
    return dests


def knight_moves(M, r, c):
    color = np.sign(piece_at(M, r, c))
    return _offset_squares(M, r, c, KNIGHT_OFFSETS, color)


def king_moves(M, r, c):
    color = np.sign(piece_at(M, r, c))
    return _offset_squares(M, r, c, KING_OFFSETS, color)


def _line_moves(M, r, c, dr, dc, color):
    """Sliding-piece ray search: the first obstruction along a direction
    is found by a prefix-sum matrix multiplication (L @ occ)."""
    steps = np.arange(1, 8)
    rs, cs = r + dr * steps, c + dc * steps
    in_bounds = (rs >= 0) & (rs < 8) & (cs >= 0) & (cs < 8)
    rs, cs = rs[in_bounds], cs[in_bounds]
    if len(rs) == 0:
        return []

    occ = (M[rs, cs] != 0).astype(int)
    L = _LOWER[: len(occ), : len(occ)]
    prefix = L @ occ                                  # matrix-vector multiplication
    first = int(np.searchsorted(prefix, 1))           # first index with a piece

    limit = first
    if first < len(occ) and np.sign(M[rs[first], cs[first]]) != color:
        limit = first + 1                             # can land on / capture the blocker
    return list(zip(rs[:limit].tolist(), cs[:limit].tolist()))


def sliding_moves(M, r, c, directions):
    color = np.sign(piece_at(M, r, c))
    dests = []
    for dr, dc in directions:
        dests += _line_moves(M, r, c, dr, dc, color)
    return dests


def rook_moves(M, r, c):
    return sliding_moves(M, r, c, ROOK_DIRS)


def bishop_moves(M, r, c):
    return sliding_moves(M, r, c, BISHOP_DIRS)


def queen_moves(M, r, c):
    return sliding_moves(M, r, c, ROOK_DIRS + BISHOP_DIRS)


def pawn_moves(M, r, c):
    """The one piece with inherently asymmetric rules, so this is where
    if/else legitimately lives -- but the forward square is still found
    with shift(), not r+1 arithmetic."""
    color = np.sign(piece_at(M, r, c))
    direction = -1 if color == WHITE else 1   # white marches toward row 0
    start_row = 6 if color == WHITE else 1
    dests = []

    src = np.zeros((8, 8), dtype=int)
    src[r, c] = 1

    one_step = shift(src, direction, 0)
    if one_step.sum() == 1:
        rr, cc = (int(x) for x in np.argwhere(one_step)[0])
        if piece_at(M, rr, cc) == 0:
            dests.append((rr, cc))
            if r == start_row:
                two_step = shift(src, 2 * direction, 0)
                rr2, cc2 = (int(x) for x in np.argwhere(two_step)[0])
                if piece_at(M, rr2, cc2) == 0:
                    dests.append((rr2, cc2))

    for dc in (-1, 1):
        diag = shift(src, direction, dc)
        if diag.sum() == 1:
            rr, cc = (int(x) for x in np.argwhere(diag)[0])
            occupant = piece_at(M, rr, cc)
            if occupant != 0 and np.sign(occupant) != color:
                dests.append((rr, cc))

    return dests
