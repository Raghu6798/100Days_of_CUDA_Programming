🧩 What is a “Tile”?

A tile is a sub-block of a matrix that a CUDA thread block loads into shared memory for faster computation.

Shared memory is much faster than global memory (hundreds of times faster).

Instead of each thread repeatedly reading global memory, threads cooperatively load a small tile into shared memory, do computations, then move to the next tile.

Think of it as “chunking” the matrix into smaller pieces that fit into shared memory.

⚡ How Tiling Works in MatMul

Global matrices: A (MxK), B (KxN) → compute C (MxN)

Each thread block handles a submatrix of C, e.g., 16×16 threads → 16×16 output tile.

For each iteration along the K dimension:

Load a tile of A and a tile of B into shared memory.

Each thread computes partial sums for its element in the C tile.

Move to the next K tile, accumulate results.

Write the final tile of C back to global memory.

🔄 Example Visualization

Suppose:

Matrix A = 32×32

Matrix B = 32×32

Thread block = 8×8 → each block computes an 8×8 tile of C

Step 1: Load 8×8 tile from A and B into shared memory.
Step 2: Each thread computes a partial sum for its element of C using these tiles.
Step 3: Move to next 8×8 tile along K dimension.
Step 4: Repeat until all partial sums are computed.
Step 5: Write final 8×8 tile of C to global memory.

✅ Why Tiling Helps

Reduces global memory accesses, which are slow.

Exploits shared memory reuse: each element of a tile is used by multiple threads.

Improves coalesced memory access and overall throughput.