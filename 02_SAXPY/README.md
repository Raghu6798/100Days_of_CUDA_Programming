# Level 1 BLAS Operations

## Characteristics of Level 1 BLAS

- Operates on **vectors only** (no matrices).  
- Performs **basic arithmetic operations** like addition, scaling, and dot products.  
- Often **memory-bound** rather than compute-bound (limited by memory bandwidth).  

## Common Level 1 BLAS Operations

### SAXPY / DAXPY
- Stands for **Single-precision A·X Plus Y (SAXPY)** or **Double-precision (DAXPY)**.  
- Computes:  

\[
y \gets \alpha x + y
\]

where \(\alpha\) is a scalar, and \(x\) and \(y\) are vectors.

### Vector Scaling (SCAL)
- Computes:  

\[
x \gets \alpha x
\]

### Dot Product (DOT)
- Computes:  

\[
x \cdot y = \sum_i x_i y_i
\]

### Vector Copy (COPY)
- Copies one vector to another:  

\[
y \gets x
\]

### Vector Norm (NRM2)
- Computes the Euclidean norm:  

\[
\|x\|_2 = \sqrt{\sum_i x_i^2}
\]

## Why SAXPY is Level 1
- It operates **element-wise on two vectors** (`x` and `y`).  
- It **doesn’t involve matrices**.  
- It’s a **memory-heavy operation**, typical of Level 1 BLAS.  

> 💡 In CUDA, implementing **SAXPY** as a kernel is often the first vector operation exercise, just like vector addition. It helps you get familiar with **thread indexing, memory access patterns, and parallelization**.
