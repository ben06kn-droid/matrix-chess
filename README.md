# Matrix Chess

A chess engine where the board state and move execution are expressed as
linear algebra operations on an 8x8 matrix — not metaphorically, but
literally: moving a piece is a Hadamard mask plus an outer-product
addition, knight/king offsets are matrix-shift products, and sliding-piece
collision detection is a triangular prefix-sum matrix multiplication.

The parts of chess that are genuinely non-linear (turn order, legality,
pawn's asymmetric rules) are left as ordinary control flow rather than
forced into a matrix that can't represent them. The point of the project
is the line between those two categories, not pretending the line
doesn't exist.

## Encoding

`M` is an 8x8 integer matrix.

- `sign(M[r,c])`: +1 white, -1 black, 0 empty
- `abs(M[r,c])`: 1=pawn, 2=bishop, 3=knight, 4=rook, 5=queen, 6=king
- row 0 = rank 8, row 7 = rank 1; col 0 = file a, col 7 = file h

## The core operations

**Extraction** is a pure matrix product: `M[r,c] = e_r^T · M · e_c`

**Moving a piece** (`apply_move` in `board.py`) is one affine update that
handles quiet moves *and* captures identically, with zero piece-specific
branching:

```
M' = (M ⊙ Z) + v · E(r2, c2)        where Z = J − E(r,c) − E(r2,c2)
```

`E(i,j) = e_i e_j^T` is an elementary matrix built from an outer product;
`Z` Hadamard-masks out the source and destination cells before the moved
piece's value is dropped back in.

**Knight and king moves** (`moves.py`) shift a one-hot "piece-only" matrix
by a fixed offset using `shift_matrix(k)` products — pieces that fall off
the board edge vanish for free because the shift matrices truncate rather
than wrap.

**Sliding pieces** (rook/bishop/queen) find their first obstruction via
`prefix = L @ occ`, a lower-triangular matrix multiplied against the
occupancy vector along a ray — the first nonzero entry is the blocker.

**Pawns** are where if/else legitimately lives: direction, the
two-square first move, and diagonal-only captures are asymmetric rules,
not facts about linear maps. Even here the forward square is found with
`shift()`, not `row + 1`.

## Run it

```bash
pip install numpy
python demo.py
```

## What's deliberately not built yet

This is move *generation* and move *execution*, not a rules-complete
legal-move filter. Missing, in roughly the order I'd add them:

- **Check detection / legal-move filtering** — a move is only legal if it
  doesn't leave your own king attacked. This requires simulating the move
  and re-running attack generation for the opponent, which is inherently
  a search, not a single matrix expression.
- **Castling, en passant, promotion** — each has its own one-off
  condition; not worth distorting the matrix formalism to absorb them.
- **Checkmate/stalemate detection.**
- A thin game loop / CLI, then maybe a small web visualizer.

## Why this representation, honestly

A straight matrix-multiplication chess engine isn't fully achievable —
legality is conditional on board state in ways a fixed linear map can't
express. What *is* true is that state representation, single-piece
translation, fixed-offset pieces, and ray collision detection all have
clean linear-algebra formulations, and isolating exactly which 80% of
the engine is linear (and why the other 20% can't be) is more
interesting than either "it's all just matrices" or "matrices are
irrelevant to chess."
