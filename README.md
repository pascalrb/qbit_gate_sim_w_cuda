# Qbit Gate Operation Parallelization with CUDA

Simulation of a single-qbit quantum gate. A single-qubit gate operation can be simulated as many 2x2 matrix multiplications. In an n-qubit quantum circuit, the n-qubit quantum state is represented as a vector of length N = 2^n, i.e.,  \[a0, a1, ..., aN-1]. The vector indices can be represented with binary notations. For example, a9 = a0001001 in a 7-qubit system.

CUDA parallelization infrastructure is used along with an NVIDIA GPU to leverage the parallelization nature of qubit gate operation which could be organized as a matrix multiplication problems.

Guided by a project from ECE786 at NC State University taught by Prof. [Huiyang Zhou](https://ece.ncsu.edu/people/hzhou/).
