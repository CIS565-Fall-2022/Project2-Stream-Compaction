CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Shixuan Fang
  * [LinkedIn](https://www.linkedin.com/in/shixuan-fang-4aba78222/)
* Tested on: Windows 11, i7-12700k, RTX3080Ti (Personal Computer)

# Overview

This project is mainly about **Parallel Scan(prefix sum)** and **Stream Compaction** algorithms implemented with CUDA. Scan is about computing the prefix sum of an array, and stream compaction is about deleting all elements in an array that meet certain condition. These algorithms seem to be inherently sequential at the first glance, but with GPU we can convert these algorithms into very efficient parallel algorithms.

One application of Parallel Scan is Summed Area Table, which is a very important algorithm real-time rendering, especially for pre-computation. Another one is Radix Sort, which is a sorting algorithm that can run in parallel. Stream Compaction is very important in ray tracing, which can help delete unnecessory rays.

# Description
In this project, I mainly implemented these algorithms:
- Naive GPU Scan & Stream compaction
- Naive GPU Scan 
- Optimized Efficient GPU Scan & Stream Compaction (which break the scanning process into up-sweep and down-sweep stages)
- GPU Scan with Thrust Library
- More-Efficient GPU Scan with dynamic number of blocks and threads (extra credit)
- GPU Scan Using Shared Memory && Hardware Optimization (extra credit)
- Radix Sort (extra credit)

# Output

```
****************
** SCAN TESTS **
****************
    [  37   0  28  30  42  12  17  24  24  11   7  27   6 ...  23   0 ]
==== cpu scan, power-of-two ====
   elapsed time: 53.0417ms    (std::chrono Measured)
==== cpu scan, non-power-of-two ====
   elapsed time: 53.5831ms    (std::chrono Measured)
    passed
==== naive scan, power-of-two ====
   elapsed time: 9.51712ms    (CUDA Measured)
    passed
==== naive scan, non-power-of-two ====
   elapsed time: 9.58666ms    (CUDA Measured)
    passed
==== work-efficient scan, power-of-two ====
   elapsed time: 8.21565ms    (CUDA Measured)
    passed
==== work-efficient scan, non-power-of-two ====
   elapsed time: 7.78179ms    (CUDA Measured)
    passed
==== optimized work-efficient scan, power-of-two ====
   elapsed time: 4.05126ms    (CUDA Measured)
    passed
==== optimized work-efficient scan, non-power-of-two ====
   elapsed time: 4.13146ms    (CUDA Measured)
    passed
==== work-efficient scan with shared memory, power-of-two ====
   elapsed time: 1.93226ms    (CUDA Measured)
    passed
==== work-efficient scan with shared memory, non power-of-two ====
   elapsed time: 1.75846ms    (CUDA Measured)
    passed
==== thrust scan, power-of-two ====
   elapsed time: 1.15917ms    (CUDA Measured)
    passed
==== thrust scan, non-power-of-two ====
   elapsed time: 0.641792ms    (CUDA Measured)
    passed

*****************************
** Radix Sort TESTS **
*****************************
==== Radix sort ====
   elapsed time: 0.641792ms    (CUDA Measured)
    passed

*****************************
** STREAM COMPACTION TESTS **
*****************************
    [   2   1   2   1   3   1   2   2   2   2   3   3   2 ...   0   0 ]
==== cpu compact without scan, power-of-two ====
   elapsed time: 58.6994ms    (std::chrono Measured)
    passed
==== cpu compact without scan, non-power-of-two ====
   elapsed time: 58.559ms    (std::chrono Measured)
    passed
==== cpu compact with scan ====
   elapsed time: 146.589ms    (std::chrono Measured)
    passed
==== work-efficient compact, power-of-two ====
   elapsed time: 11.0158ms    (CUDA Measured)
    passed
==== work-efficient compact, non-power-of-two ====
   elapsed time: 10.992ms    (CUDA Measured)
    passed
```


# Performance Analysis

- Scan runtime with different block size with array size = 2^25.

![Performance with different block size (1)](https://user-images.githubusercontent.com/54868517/190933826-32efa973-fad8-4ba4-9508-5a614bd00540.png)


As seen in this graph, **blockSize = 256** has the best performance for Naive, Efficient, and Efficient with shared memory, therefore these blockSize are set to **256**, **blockSize = 128** has the best performance for Optimized Efficient, which is then set to **128**


- Compare all of these GPU Scan implementations to CPU version of Scan 

![17d41e321d148779ee125d48673dcc7](https://user-images.githubusercontent.com/54868517/190935115-3d188d84-bad3-4d8c-b0e6-bf09634840b1.jpg)

- In this graph, array size starts with 2^20 and ends with 2^28; generally, GPU scan algorithms are faster than CPU serial scan with array size larger than 2^16 due to great amount of parallel computing.
- As seen clearly in this graph, other than thrust library, Efficient scan with shared memory performed the best, which is much faster than CPU scan, even cost only half time compared with Optimized Efficient scan. This implies that **global memory accessing** is one of the biggest performance bottleneck.
- We also noticed that the Optimized Efficient Scan(with dynamic block numer/grid size) outperform the naive Efficient Scan. This implies that **the number of blocks that are launched in a kernel do affect performance**. When we update the number of blocks in Up-Sweep stage and Down-Sweep stage dynamically, the kernels can run faster.
- Thust::exclusive_scan is always the fastest method, which is even much master than Efficient Scan with shared memory.
 - As seen in the following screenshot from Nsight Timeline, the thrust api has ```cudaMemcpyAsync()``` and ``` cudaStreamSynchronize``` functions which are not used in my implementation. According to a stackoverflow post, ```cudaMemcpyAsync()``` will **returns control to the host immediately (just like a kernel call does) rather than waiting for the data copy to be completed.** I assume this will greatly improve the performance since we can do something else while cpu is copying data.
![be2a7736d65f0a09e21ce97efc51d27](https://user-images.githubusercontent.com/54868517/190936662-7465d555-925c-4a09-aa21-9e67cc1f1aea.jpg)

# Extra Credit

**1. Optimized GPU Efficient Scan**
```
__global__ void kernUpSweep(int n, int stride, int* data)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n) return;
    if (index % stride != 0 || index + stride -1  >= n) return;
    data[index + stride - 1] += data[index + stride / 2 - 1];
}
```
For both kernUpSweep and kernDownSweep kernels used in the basic scan function, there are ```%``` operations, which are very slow to compute in GPU. After optimization, the new KernOptimizedUpSweep looks like this:
```
__global__ void kernOptimizedUpSweep(int n, int d, int offset, int* data)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n) return;
    if (index < d) {
        int ai = offset * (2 * index + 1) - 1;
        int bi = offset * (2 * index + 2) - 1;
        data[bi] += data[ai];
    }
}
```
This simple change actually improved the performance a lot. 
Another part I've changed is how many blocked are launched, this is achieved by shrinking block number by two in the for loop during up sweep stage, and expand by two during the down sweep stage.
```
for (int d = Num >> 1; d > 0; d >>= 1) {
    blockNum = (d + blockSize - 1) / blockSize;
    kernOptimizedUpSweep << <blockNum, blockSize >> > (Num, d, offset, dev_data);
    offset <<= 1;
}
```

**2. GPU Efficient Scan with shared memory**

This is mainly achieve by the kernal ```kernOptimizedPerBlockScan```, which is adapted from GPU Gem 3 Ch 39. 
The source code in GPU Gem 3 only works with 1 block, so I changed it to allow multiple blocks run this algorithm in parallel and then implemented the algorithm showed in class slide.

**3. Radix Sort**

I've also implemented Radix Sort, which can be see in ```StreamCompaction::Efficient::RadixSort()```

Here is an example of how I tested it.
```
** Radix Sort TESTS **
*****************************
    [  43  34  35  24  32   8  19   0 ]
==== Radix sort ====
    [   0   8  19  24  32  34  35  43 ]
==== Result from std::sort ====
    [   0   8  19  24  32  34  35  43 ]
   elapsed time: 0.493472ms    (CUDA Measured)
    passed
```
