# GPU Programming - CUDA & GPU Kernels

This repository showcases the evolution from basic linear algebra operations to advanced machine learning algorithms, all optimized for parallel execution on GPUs.

## 📋 Project Overview

This project contains **100+ GPU kernels** implementing a wide range of algorithms and operations:
- **CUDA** (NVIDIA GPUs) - Primary implementation language
- **HIP** (AMD GPUs via ROCm) - Portable GPU implementations
- **Triton** - High-level GPU programming abstractions
- **cuBLAS/cuDNN** - Optimized library integrations
---

## 📁 Repository Structure

### Core Categories

#### 1. **linear_algebra/** - Vector & Matrix Operations
Fundamental building blocks for GPU computing including matrix multiplication, transpose operations, and specialized algorithms.

**Key Files:**
- `vectAdd.cu` - Vector addition kernel
- `MatrixAdd.cu` - Element-wise matrix addition
- `matrix_vec_mult.cu` - Matrix-vector multiplication
- `MatrixTranspose.cu` - Optimized transpose with shared memory
- `vectorAdditionCublas.cu`, `MutMul_cublas.cu` - cuBLAS library usage
- `sparse_MatrixVecMult_Hybrid_ELL_COO.cu` - Sparse matrix operations
- `sgemm.cpp`, `strassen.cpp`, `winograd.cpp` - Advanced GEMM algorithms
- Various optimizations for different GPU architectures

#### 2. **convolution/** - Convolution Operations
1D, 2D, and 3D convolution implementations with memory optimization techniques.

**Key Files:**
- `one_d_convolution.cu` - Basic 1D convolution
- `one_d_convolution_with_tiling.cu` - Optimized with tiling
- `2d_convolution_with_tiling.cu` - 2D convolution with halo cells
- Efficient memory coalescing and shared memory management

#### 3. **attention/** - Transformer & Attention Mechanisms
Memory-efficient attention implementations including Flash Attention and rotary positional embeddings.

**Key Files:**
- `flash_attention_forward.cu` - Fast attention forward pass
- `flash_attention_backprop.cu` - Gradient computation for attention
- `RoPE.cu` - Rotary positional embeddings
- `rope_hip.cpp` - HIP version for AMD GPUs
- `mrope.cu` - Multi-modal rotary embeddings

#### 4. **ml_kernels/** - Activation Functions & Normalization
Low-level ML operations including layer normalization and various activation functions.

**Key Files:**
- `LayerNorm.cu` - Layer normalization with numerical stability
- `GELU.cu`, `geglu.cu` - Gaussian and gated activation functions
- `SwiGLU.cu` - Swish-gated linear unit
- `group_norm.cu`, `RMS_Normalization.cu` - Alternative normalizations
- `mish.cu`, `Softplus.cu`, `hard_sigmoid.cu`, `ELU.cu`, `leaky_relu.cu`

#### 5. **neural_networks/** - Deep Learning Layers & Networks
Complete implementations of neural network layers and architectures.

**Key Files:**
- `fcnet.cu` - Fully connected networks
- `cnn.cu` - Convolutional neural networks with forward/backward passes
- `lstm.cu`, `lstm_bidirectional.cu` - LSTM and bidirectional variants
- `2D_Max_Pooling.cu`, `3d_average_pooling.cu` - Pooling operations
- `lora.cu` - Low-rank adaptation for efficient fine-tuning
- `SN.cu` - Spectral normalization for GANs

#### 6. **ml_algorithms/** - Optimization & Loss Functions
Machine learning algorithms including optimizers and various loss functions.

**Key Files:**
- Optimizers: `Muon.cu`, `lbfgs.cu`, `CGM.cu`, `AdaHessian.cu`
- Loss Functions: `kl_divergence.cu`, `contrastive_loss.cu`, `triplet_loss.cu`, `Huber_Loss.cu`, `MSE.cu`
- Clustering: `k_means_leetgpu.cu`
- Classifiers: `NaiveBayes.cu`
- Advanced: `EM_kernel.cu`, `jsd_cuda.cu` (Jensen-Shannon Divergence)

#### 7. **sorting/** - Sorting & Reduction Algorithms
Parallel sorting and prefix sum operations for efficient data processing.

**Key Files:**
- `merge_sort.cu` - GPU-optimized merge sort
- `prefixsum_brent_kung_algorithm.cu` - Work-efficient prefix sum
- `partial_sum.cu` - Parallel reduction operations
- Bitonic sort and parallel merge implementations

#### 8. **graph_algorithms/** - Graph Operations
Graph traversal and manipulation algorithms.

**Key Files:**
- `BFS/` - Breadth-first search implementations
- Graph processing kernels with efficient memory access patterns

#### 9. **signal_processing/** - Signal & FFT Operations
Fourier transforms and signal processing on GPUs.

**Key Files:**
- FFT implementations
- Signal processing kernels

#### 10. **scientific_computing/** - Numerical Methods
Physics simulations and scientific computing algorithms.

**Key Files:**
- Poisson solver
- Numerical integration methods
- Finite difference schemes

#### 11. **simulation/** - Physics & Cellular Automata
Interactive simulations and cellular automaton implementations.

**Key Files:**
- `game_of_life.cu` - Conway's Game of Life
- `three_body_problem.cu` - N-body simulation
- Heat diffusion and fluid dynamics simulations
- Boids flocking behavior

#### 12. **utilities/** - Helper Functions & Miscellaneous
Support code and utility kernels.

**Key Files:**
- General-purpose GPU utilities
- Benchmark code and test utilities

---

## 🚀 Quick Start

### Prerequisites

- **NVIDIA GPU** (CUDA 11.0+) or **AMD GPU** (ROCm compatible)
- **CUDA Toolkit** or **ROCm** installed
- C++ compiler (MSVC, GCC, or Clang)
- CMake (optional, for build management)

### Compilation Examples

#### Single CUDA File
```bash
nvcc -o kernel kernel.cu
./kernel
```

#### With Optimization Flags
```bash
nvcc -O3 -arch=sm_80 -o kernel kernel.cu
```

#### HIP Version (AMD GPUs)
```bash
hipcc -O3 -o kernel kernel.cpp
```

#### Python Integration
```bash
# Requires PyTorch/CuPy
python torch_test.py
```
---


## 🎯 Key Algorithms Implemented

### Linear Algebra
- **Matrix Multiplication** - Multiple optimization strategies (register tiling, Winograd, Strassen)
- **Vector Operations** - SAXPY, dot product, norms
- **Sparse Matrices** - CSR, COO, ELL formats
- **Decompositions** - LU, QR (via cuBLAS)

### Machine Learning
- **Attention** - Flash Attention (memory-efficient), RoPE positional embeddings
- **Normalizations** - Layer Norm, Group Norm, RMS Norm
- **Activations** - ReLU, GELU, Swish, Mish, ELU, and variants
- **Loss Functions** - Cross-entropy, Contrastive, Triplet, KL Divergence
- **Optimizers** - Adam (variant: Muon), L-BFGS, Conjugate Gradient, AdaHessian

### Algorithms
- **Sorting** - Merge Sort, Bitonic Sort, Quicksort variants
- **Prefix Operations** - Brent-Kung, Blelloch scan
- **Graph Traversal** - BFS, DFS
- **Clustering** - K-means, EM algorithm

### Physics & Simulation
- **N-body Problem** - Efficient force calculation
- **Game of Life** - Cellular automaton
- **Fluid Dynamics** - Heat diffusion, Navier-Stokes
- **Wave Functions** - Quantum mechanics simulations

---

## 💡 Best Practices & Optimization Techniques

### Memory Optimization
- **Shared Memory Tiling** - Reduce global memory access
- **Coalesced Access** - Align memory patterns for bandwidth
- **Register Blocking** - Maximize register usage
- **Halo Cells** - Efficient boundary handling in stencil operations

### Computation
- **Fused Operations** - Combine kernels to reduce memory traffic
- **Work Stealing** - Balance load across warps/blocks
- **Warp Shuffle** - Efficient intra-warp communication
- **Atomic Operations** - When necessary for correctness

### Portability
- **HIP Implementations** - Run on both NVIDIA and AMD
- **Triton Abstractions** - High-level kernel programming
- **Library Integration** - cuBLAS/cuDNN for standard operations

---

## 📈 Performance Considerations

Each kernel includes:
- Optimal block/grid configurations for the operation
- Memory access pattern optimization
- Register usage minimization
- Bank conflict avoidance in shared memory

Benchmarking code is provided in `day 11/benchmark.py` for performance comparison across different implementations.

---

## 📚 References & Resources

### CUDA Programming
- [NVIDIA CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/)
- [NVIDIA GPU Computing Gems](https://developer.nvidia.com/gpu-computing-gems)

### Algorithm References
- Flash Attention: [Dao et al., 2022](https://arxiv.org/abs/2205.14135)
- Rotary Embeddings (RoPE): [Su et al., 2021](https://arxiv.org/abs/2104.09864)
- Efficient Attention: [Rabe & Turban, 2023](https://arxiv.org/abs/2307.08691)

### HIP Programming
- [AMD ROCm Documentation](https://rocmdocs.amd.com/)
- [HIP Programming Guide](https://rocmdocs.amd.com/en/docs-5.0.0/Programming_Guides/HIP-GUIDE.html)

---

## 💻 Platform Support

| Platform | Language | Status |
|----------|----------|--------|
| NVIDIA GPU | CUDA | ✅ Primary |
| AMD GPU | HIP | ✅ Supported (Days 30-33) |
| CPU Comparison | C++ | ✅ Available (Day 30) |
| High-Level | Triton | ✅ Available (Days 38-43, 58-61) |

---

## 📝 License

This project is provided as educational material for GPU programming learning and research purposes.

---

## 🤝 Contributing

Found optimizations or new algorithms? This project welcomes improvements and additional implementations.

### Adding a New Kernel
1. Choose the appropriate category folder
2. Follow the naming convention: `algorithm_name.cu` or `.cpp`
3. Include performance comments
4. Add brief algorithm description in file header

---

## 📧 Contact & Support

For questions or suggestions about the GPU programming implementations, please refer to the individual kernel files for detailed comments and algorithm explanations.

---

**Last Updated:** January 2026  
**Total Kernels:** 100+  