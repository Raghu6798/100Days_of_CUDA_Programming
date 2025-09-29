# 100 Days of CUDA Programming

Welcome to my **100 Days of CUDA Programming** repository! 🚀  

This repo documents my journey of learning **CUDA programming** with a focus on writing custom GPU kernels to accelerate computations. My aim is to leverage these skills as an **LLM Engineer** to optimize inference and experiment with efficient GPU operations for large language models and deep learning workloads.

---

## Why CUDA?

CUDA (Compute Unified Device Architecture) is **NVIDIA's parallel computing platform** that allows developers to write programs that run on GPUs. By writing **custom kernels**, I can speed up matrix operations, vector operations, and other heavy computations that are critical in AI and LLM pipelines.

---

## Repo Structure

Each day’s exercises are organized in separate folders:

day_1/ # Vector addition
day_2/ # SAXPY operation
day_3/ # Dot product kernel
...
common/ # Utility headers and includes


Typical files in each day:

- `src/` : Contains `.cpp`, `.cu`, and test scripts
- `setup.py` : Builds C++/CUDA extensions
- `Makefile` : Automates build and test commands
- `test.py` : Python script to verify kernel operations

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/100Days-of-CUDA-Programming.git
cd 100Days-of-CUDA-Programming
```

2. Set up a Python virtual environment (optional but recommended)
``` bash
python -m venv .venv
```
# Windows
``` bash
.venv\Scripts\activate
```
# Linux/macOS
```bash
source .venv/bin/activate
 ```  bash   
pip install --upgrade pip
pip install torch setuptools ninja
```

Make sure your machine has an NVIDIA GPU and the correct CUDA toolkit installed.

3. Build and run the CUDA kernels

Each day folder contains a Makefile and a setup.py. You can build and test your kernels using Make:


# Build and test
make all

# OR build only
make build

# Run tests only
make run

# Clean build artifacts
make clean

make build → Compiles the CUDA/C++ extension in-place

make run → Runs the test Python script

make all → Builds and runs tests automatically

make clean → Removes all compiled and temporary files

## My Learning Goal

As an LLM Engineer, the goal of this 100-day journey is to:

# 1)Gain a deep understanding of GPU parallelism and memory management.
# 2)Write custom CUDA kernels to accelerate operations like vector addition, SAXPY, dot products, and matrix multiplications.
# 3)Experiment with optimizations for LLM inference, such as efficient tensor computations and memory access patterns.
# 4) Build a foundation for integrating low-level GPU acceleration in deep learning pipelines.

### Contributions

This repository is primarily for personal learning. However, contributions in the form of:

1)Optimized CUDA kernels

2)Additional exercises

3) Notes or documentation improvements

are welcome via pull requests.

### License

This repository is licensed under the MIT License. See the LICENSE
 file for details.

Happy CUDA programming! ⚡️


I can also make a **Day-wise table of contents** with all 100 days listed in Markdown so readers see the roadmap clearly.  

Do you want me to create that too?


