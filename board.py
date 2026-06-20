"""
Matrix-based chess board representation.

Encoding
--------
Board state is an 8x8 integer matrix M.
    sign(M[r, c])  -> color: +1 white, -1 black, 0 empty
    abs(M[r, c])   -> piece type:
        1 = pawn, 2 = bishop, 3 = knight,
        4 = rook, 5 = queen, 6 = king

Squares: row 0 = rank 8 (black's back rank), row 7 = rank 1 (white's back rank).
         col 0 = file a, col 7 = file h.
"""

import numpy as np

EMPTY, PAWN, BISHOP, KNIGHT, ROOK, QUEEN, KING = range(7)
PIECE_LETTERS = {PAWN: 'P', BISHOP: 'B', KNIGHT: 'N', ROOK: 'R', QUEEN: 'Q', KING: 'K'}
WHITE, BLACK = 1, -1


def initial_board() -> np.ndarray:
    """Return the standard starting position as an 8x8 int matrix."""
    back_rank = np.array([ROOK, KNIGHT, BISHOP, QUEEN, KING, BISHOP, KNIGHT, ROOK])
    M = np.zeros((8, 8), dtype=int)
    M[0, :] = -back_rank   # black back rank (row 0 = rank 8)
    M[1, :] = -PAWN        # black pawns
    M[6, :] = PAWN         # white pawns
    M[7, :] = back_rank    # white back rank
    return M


# ---------------------------------------------------------------------
# Pure linear-algebra primitives
# ---------------------------------------------------------------------

def e(i: int) -> np.ndarray:
    """i-th standard basis column vector in R^8."""
    v = np.zeros((8, 1), dtype=int)
    v[i, 0] = 1
    return v


def E(i: int, j: int) -> np.ndarray:
    """Elementary matrix with a 1 at (i, j): the outer product e_i e_j^T."""
    return e(i) @ e(j).T


def piece_at(M: np.ndarray, r: int, c: int) -> int:
    """Extract M[r, c] via pure matrix multiplication: e_r^T M e_c."""
    return int((e(r).T @ M @ e(c)).item())


_ONES = np.ones((8, 8), dtype=int)


def apply_move(M: np.ndarray, src: tuple, dst: tuple) -> np.ndarray:
    """
    Move whatever is on `src` to `dst`, capturing anything that's there.

        M' = (M (.) Z) + v * E(dst)

    Z zeroes the source and destination cells (Hadamard mask); v*E(dst)
    places the moved piece via an outer-product matrix. The same affine
    update handles quiet moves AND captures -- no piece-specific
    branching happens in this function at all.
    """
    r, c = src
    r2, c2 = dst
    v = piece_at(M, r, c)
    Z = _ONES - E(r, c) - E(r2, c2)
    return (M * Z) + v * E(r2, c2)


def shift_matrix(k: int) -> np.ndarray:
    """
    8x8 matrix S_k such that (S_k @ x)[i] = x[i-k] if in bounds else 0.
    Composing S_k (rows) with S_k^T (cols) translates a piece by a fixed
    offset and naturally drops it if it falls off the board.
    """
    Sk = np.zeros((8, 8), dtype=int)
    for i in range(8):
        j = i - k
        if 0 <= j < 8:
            Sk[i, j] = 1
    return Sk


def shift(M: np.ndarray, dr: int, dc: int) -> np.ndarray:
    """Translate every entry of M by (dr, dc), zero-filling off-board cells."""
    return shift_matrix(dr) @ M @ shift_matrix(dc).T


def render(M: np.ndarray) -> str:
    """Pretty-print the board."""
    rows = []
    for r in range(8):
        cells = []
        for c in range(8):
            v = M[r, c]
            if v == 0:
                cells.append('.')
            else:
                letter = PIECE_LETTERS[abs(v)]
                cells.append(letter if v > 0 else letter.lower())
        rows.append(f"{8 - r}  " + ' '.join(cells))
    rows.append("   a b c d e f g h")
    return '\n'.join(rows)
