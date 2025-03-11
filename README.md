
# **KernelLab**  
**High-Performance CUDA Kernels for Deep Learning & HPC**  
Making GPUs go Brrrrr....
![LOGO](https://github.com/AmanSwar/KernelLab/blob/master/logo.png)
![CUDA](https://img.shields.io/badge/CUDA-Optimized-green?style=for-the-badge&logo=nvidia)  
![C++](https://img.shields.io/badge/C%2B%2B-17%2B-blue?style=for-the-badge&logo=c%2B%2B)  

## **About KernelLab**  
**KernelLab** is a collection of **highly optimized CUDA kernels** designed for deep learning, high-performance computing (HPC), and general-purpose GPU acceleration. Each kernel includes multiple levels of optimization—from **naïve implementations** to **shared memory, warp-level, vectorized, and tensor-core optimized** versions.  

---

## **🛠️ Features**  
- **Optimized CUDA kernels** for **deep learning, matrix operations, and image processing**  
- **Multiple optimization techniques**: Shared Memory, Coalesced Memory Access, Warp-Level Parallelism, Tensor Cores  
- **Benchmark comparisons against cuBLAS, cuDNN, and PyTorch CUDA kernels**  
- **Currently Optimized for Ampere architecture (cuz I am GPU poor)**  

---

## **📌 Implemented Kernels & Optimizations**  

### **🔹 Convolution Kernels**  
| Kernel  | Optimization Levels |  
|---------|--------------------|  
| **2D Convolution (Conv2D)** | 1️⃣ **Naïve (Direct element-wise computation)** <br> 2️⃣ **Tiled Shared Memory (Minimizing global memory access)** <br> 3️⃣ **Memory Coalescing (Optimized memory access patterns)** <br> 4️⃣ **Tensor Cores (Using WMMA for fused matrix multiplications)** |  
| **3D Convolution (Conv3D)** | 1️⃣ **Naïve** <br> 2️⃣ **Shared Memory (Minimizing redundant loads)** <br> 3️⃣ **Tiled (Reducing register pressure)** <br> 4️⃣ **Register Blocking (Efficient memory reuse via registers)** |  

### **🔹 Matrix & Reduction Operations**  
| Kernel  | Optimization Levels |  
|---------|--------------------|  
| **Matrix Transpose** | 1️⃣ **Naïve (Direct row-column swaps)** <br> 2️⃣ **Shared Memory Tiling (Blocking memory accesses for fewer global loads)** <br> 3️⃣ **Memory Coalescing (Optimizing global memory writes for aligned access)** |  
| **Matrix Multiplication (GEMM)** | 1️⃣ **Naïve (Row-major computation)** <br> 2️⃣ **Tiled (Using shared memory for efficient blocking)** <br> 3️⃣ **Register Blocking (Reducing register pressure & maximizing reuse)** <br> 4️⃣ **Warp-Level Tiling (Optimizing warp-level data exchange)** <br> 5️⃣ **Tensor Cores with WMMA (Using NVIDIA Tensor Cores for fused matrix ops)** |  
| **Reduction Sum** | 1️⃣ **Naïve (Basic sequential reduction per thread block)** <br> 2️⃣ **Branchless Reduction (Avoiding thread divergence for performance gain)** <br> 3️⃣ **Warp-Level Reduction (Using shuffle intrinsics for direct register exchange)** |  

### **🔹 Element-wise & Activation Functions**  
| Kernel  | Optimization Levels |  
|---------|--------------------|  
| **ReLU Activation** | 1️⃣ **Naïve (Basic element-wise ReLU application)** <br> 2️⃣ **Coalesced Memory Access (Optimized read/write for better bandwidth usage)** <br> 3️⃣ **Vectorized Execution (Processing multiple elements per thread using vector types like `float4`)** |  
| **SoftMax Function** | 1️⃣ **Naïve (Computing exponentials & normalizing sequentially)** <br> 2️⃣ **Shared Memory Optimization (Avoiding redundant memory accesses)** <br> 3️⃣ **Block Tiling (Parallelizing exponentiation & normalization)** <br> 4️⃣ **Warp-Level Reduction (Efficient sum-reduction across warps)** <br> 5️⃣ **State-of-the-Art Optimization (Optimized numerical stability & memory efficiency)** |  
| **Vector Addition** | 1️⃣ **Naïve (Thread-per-element)** <br> 2️⃣ **Shared Memory Optimization (Minimizing redundant memory loads)** <br> 3️⃣ **Tiled Execution (Using block-level parallelism for efficiency)** <br> 4️⃣ **Coalesced Memory Access (Optimizing memory loads for aligned access)** <br> 5️⃣ **Vectorized Computation (Using `float4` for processing multiple elements per thread)** <br> 6️⃣ **Multi-Element Processing (Reducing loop overhead for large arrays)** |  

### **🔹 Image Processing Kernels**  
| Kernel  | Optimization Levels |  
|---------|--------------------|  
| **Greyscale Conversion** | 1️⃣ **Naïve (Direct pixel-wise computation)** <br> 2️⃣ **Shared Memory Optimization (Reducing redundant loads per thread block)** <br> 3️⃣ **Memory Coalescing (Ensuring aligned memory accesses for better bandwidth)** <br> 4️⃣ **Vectorized Computation (`uchar4` processing per thread)** <br> 5️⃣ **Multi-Pixel Processing (Parallel processing of multiple pixels per thread)** <br> 6️⃣ **Fused Multiply-Add (FMA) Optimization (Combining operations for fewer instructions)** |  
| **Image Blurring** | 1️⃣ **Naïve (Basic kernel filter computation per pixel)** <br> 2️⃣ **Optimized Shared Memory Tiling (Minimizing global memory accesses by loading tiles into shared memory)** |  

---

## **📝 Currenlty implementing / TODO & Future Plans**  
- [ ] **Self-Attention CUDA Kernel**  
- [ ] **Flash Attention Kernel Optimization**  
- [ ] **LeakyReLU Kernel**  
- [ ] **Layer Normalization CUDA Kernel**  
- [ ] **FFT, BFS, DFS, and Sorting CUDA Implementations**  

---

## **📜 License**  
KernelLab is **open-source** under the **MIT License**.  

---

## **🤝 Contributing**  
PRs and issues are welcome! If you have an optimization idea or find a bug, feel free to contribute. 🚀  

---

### **💬 Contact**  
For discussions & suggestions, open an issue or DM me on GitHub!  

---

### **🔥 KernelLab: Pushing CUDA Performance to the Next Level! 🔥**  

---
