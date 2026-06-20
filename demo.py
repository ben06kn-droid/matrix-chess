from board import initial_board, render, apply_move
from moves import knight_moves, rook_moves, pawn_moves, bishop_moves

M = initial_board()
print("Starting position:")
print(render(M))

print("\nWhite knight on b1 (row 7, col 1) can go to:", knight_moves(M, 7, 1))
print("White rook on a1 (row 7, col 0) can go to (boxed in):", rook_moves(M, 7, 0))
print("White pawn on a2 (row 6, col 0) can go to:", pawn_moves(M, 6, 0))

# Push the a-pawn two squares using the masked-shift move primitive
M = apply_move(M, (6, 0), (4, 0))
print("\nAfter white plays a2-a4:")
print(render(M))
print("Rook on a1 can now go to:", rook_moves(M, 7, 0))
print("Pawn on a4 can go to (no double-step anymore):", pawn_moves(M, 4, 0))

# Knight hops over everyone -- no blocking logic involved
M = apply_move(M, (7, 1), (5, 2))
print("\nAfter Nb1-c3:")
print(render(M))
print("Knight on c3 can go to:", knight_moves(M, 5, 2))

# Force a capture to prove apply_move handles it with zero special-casing
print("\nForced-capture demo (not a legal chess move, just proving the math):")
M2 = apply_move(M, (5, 2), (1, 3))  # knight 'teleports' to grab the d7 pawn
print(render(M2))
print("Captured square now holds:", M2[1, 3], "(was black pawn = -1)")
