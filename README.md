CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Chang Liu
  * [LinkedIn](https://www.linkedin.com/in/chang-liu-0451a6208/)
  * [Personal website](https://hummawhite.github.io/)
* Tested on personal laptop:
  - Windows 11
  - i7-12700 @ 4.90GHz with 16GB RAM
  - RTX 3070 Ti Laptop 8GB

## Features

This project works on implementation and optimization of several parallelized scan-relevant algorithms with CUDA. A brief introduction of what these algorithms do and what implementations I have done:

- Scan: calculate the prefix sum (using arbitrary operator) of an array

  - [1] CPU scan with/without simulating parallelized scan (Part 1)

  - [2] GPU naive scan (Part 2)

  - [3] GPU work-efficient scan (Part 3.1 + Part 5)

  - [4] GPU work-efficient scan with shared memory optimization & bank conflict reduction (Part 7)

  - [5] GPU scan using `thrust::exclusive_scan` (Part 4)

- Stream compaction: remove elements that unmeet specific conditions from an array, and keep the rest compact in memory

  - [6] CPU stream compaction with/without CPU scan (Part 1)

  - [7] GPU stream compaction using [3] work-efficient scan (Part 3.2)

  - [8] GPU stream compaction using [4] optimized work-efficient scan (Part 7)

- Radix sort

  - [9] GPU radix sort using [3] work-efficient scan (Part 6)

  - [A] GPU radix sort using [4] optimized work-efficient scan (Part 7)

### Brief Overview of Extra-Credit Implementations

#### [3] Work-Efficient Scan Not Slower than CPU

I knew that after each iteration of partial upsweep/downsweep the number of active threads halves and keeping them compact would significantly reduce control divergence, so I optimized it at the very beginning. Then it was faster than CPU.

#### [4] Work-Efficient Scan Optimized with Shared Memory

To utilize shared memory, we need to consider block-wisely. As shown in the picture [[GPU Gems 3 Figure 39-6](https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda)], the idea can be divided into three steps:

<div align="center"><image src="https://developer.nvidia.com/sites/all/modules/custom/gpugems/books/GPUGems3/elementLinks/39fig06.jpg" /></div>

- First, perform scan on each block and store the sum of each block into `blockSum` array
- Then, scan `blockSum` to calculate offset to be added to each block
- Last, add offset back to each block

In the second step, however, the size of `blockSum` array isn't always less than block size when number of blocks is larger than block size. In this case, to scan `blockSum`, we have two options:

- Call [3] Work-Efficient Scan to scan
- Recursively, scan `blockSum` block-wisely and store the sum of each block into another array. Repeat this step until the size of array we currently scan is less than a block
  - Similarly, to get the final result, we recursively add back block sums

It turned out that the second approach was about 2.5% faster, though it was much more complex to implement. I even wrote a class `DevSharedScanAuxBuffer` to manage the memory required to perform recursion.

##### Bank Conflict Reduction

According to [[GPU Gems 3 Chap. 39.2.3](https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda)], adding a small trick to address calculation leads to way better bank access pattern and reduces shared memory access time.

Even though GPU Gems has its code:

```C++
#define CONFLICT_FREE_OFFSET(n)
	\ ((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS)) 
```

 I doubt if it really works. So I wrote my own:

```C++
__device__ inline int conflictFreeAddr(int idx) {
	return idx + (idx >> 5);
}
```

#### [9, A] Radix Sort

##### Implementation

Basically, my implementation followed the idea shown by this picture [[GPU Gems 3 Figure 39-14](https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda)]

<div align="center">
    <image src="https://developer.nvidia.com/sites/all/modules/custom/gpugems/books/GPUGems3/elementLinks/39fig14.jpg" width="50%"/>
</div>

During implementation, we don't really need auxiliary buffers other than f. The steps above can be combined into three kernels (see [`stream_compaction/radixsort.cu`](./stream_compaction/radixsort.cu)):

- `kernMapToBool` that maps input to `e`
- `kernScan`, our GPU scan kernel, produces `f`
- `kernScatter` that computes `t` and `d`, and scatters input to final result

##### Usage

Two files are added for radix sort implementation, [`stream_compaction/radixsort.h`](./stream_compaction/radixsort.h) and [`stream_compaction/radixsort.cu`](stream_compaction/radixsort.cu).

To use radix sort, call either `StreamCompaction::RadixSort::sort` (based on work-efficient scan) or `StreamCompaction::RadixSort::sortShared` (based on shared memory optimized scan).

## Performance Analysis

### Choosing Block Size (to answer Q1)

#### Dynamic Block Size for [2] Naive Scan & [3] Work-Efficient Scan

My GPU naive scan and work-efficient scan without shared memory are coded as multi-pass algorithms. During each pass the algorithms perform partial scan/sweep over the array, and the number of threads to be launched halves/doubles by each iteration. In other words, the number of threads is dynamic.

So for these two, I actually didn't use a fixed block size. Instead, I let the block size change depending on **data amount**, **number of SMs** and **max block size**. Here is the code (in `stream_compaction/common.h`):

```C++
inline int getDynamicBlockSizeEXT(int n) {
    return std::max(warpSize(), std::min(maxBlockSize(), floorPow2(n / numSM())));
}
```

My explanation:

- When data amount is small, use smaller blocks to spread the workload onto as many SMs as possible
- When data amount is large, use larger blocks

#### The Optimal Block Size for [4] Optimized Work-Efficient Scan (to answer Q1)

Since this implementation is using shared memory and the size of shared memory required by each block is proportional to the number of threads in that block, the block size is not supposed to be large.

I tested block sizes from 32 to 1024, and discovered:

- When `block size >= 256`, explicit declaration of the size of shared memory is required in the kernel or it would crash on launch
- `block size = 128` gives the best performance, as shown in the graph below

<div align="center">
    <image src="./img/scan_shared_time_block_size.png" width="65%">
</div>

What makes it more interesting is that when I used NSight to peek how `thrust::exclusive_scan` worked, it also launched the kernel `DeviceScanKernel` with block size of 128.

### Comparison of Different Implementations (to answer Q2)

- Million Array Elements Per Second: array size / (execution time * 10^6). Inspired by MIPS, this reflects algorithms' scalability with growing amount of data and makes it more straightforward to compare different algorithms

#### Scan

<div align="center">
    <image src="./img/scan_time.png" />
    <image src="./img/scan_maeps.png" />
</div>

From the graphs above we can see how optimizing our algorithm step by step finally makes huge difference:

- The naive scan is about twice fast as CPU scan
- The work-efficient scan is 2.5~4 times fast as naive scan
- After shared memory optimization, the work-efficient scan gains more than 300% improvement, and reaches about 40% performance of Thrust's implementation

##### What Does Thrust Do

By profiling with NSight Computing, three kernels are found to be possibly associated with `thrust::exclusive_scan`:

- `_kernel_agent`
- `DeviceScanInitKernel`
- `DeviceScanKernel`

[Some related code I found in `NVIDIA/cub`](https://github.com/NVIDIA/cub/blob/main/cub/warp/specializations/warp_scan_shfl.cuh)

##### The Effect of Reducing Bank Conflict

Tested on [4] shared memory optimized scan.

<div align="center">
    <image src="./img/bank_maeps.png" />
</div>

After reducing bank conflict, the performance increases about 10%.

#### Stream Compaction

<div align="center">
    <image src="./img/compaction_time.png" />
    <image src="./img/compaction_maeps.png" />
</div>

This gives a similar result to scan algorithm.

#### Radix sort

I used `std::sort` for CPU sort as a baseline to compare my GPU radix sort with.

I discovered that the performance of CPU sort depends on the distribution of input data. To make it more general, I did all tests with random `int32`s which range from `[0, INT_MAX - 1]`.

<div align="center">
    <image src="./img/sort_time.png" />
    <image src="./img/sort_maeps.png" />
</div>

Thanks to well-optimized GPU scan, my GPU radix sort has about 10 times performance as CPU sort. 

### Bottlenecks of Different Implementations (to answer Q3)

#### Naive Scan

Definitely global memory access. The original algorithm requires $O(n \log{n})$ memory access.

#### Work-Efficient Scan

Global memory access & idle threads.

#### Work-Efficient Scan with Shared Memory

I think it's mainly due to idle threads. Even though the amount of calculation is $O(n)$, the algorithm's nature forces the process to be partly sequential, with on average blockSize / log(blockSize) threads active in each loop.

### Test Program Output Example (to answer Q4)

I added additional tests for [4, 8, 9 and A]. This example was tested with `SIZE = 1 << 28`.

```
****************
** SCAN TESTS **
****************
    [   0  40  41  32  23  15  34  16  45  12  28  40  45 ...  36   0 ]
==== cpu scan, power-of-two ====
   elapsed time: 591.764ms    (std::chrono Measured)
    [   0   0  40  81 113 136 151 185 201 246 258 286 326 ... -2015488848 -2015488812 ]
==== cpu scan, non-power-of-two ====
   elapsed time: 601.103ms    (std::chrono Measured)
    [   0   0  40  81 113 136 151 185 201 246 258 286 326 ... -2015488882 -2015488854 ]
    passed
==== naive scan, power-of-two ====
   elapsed time: 306.289ms    (CUDA Measured)
    [   0   0  40  81 113 136 151 185 201 246 258 286 326 ... -2015488848 -2015488812 ]
    passed
==== naive scan, non-power-of-two ====
   elapsed time: 306.206ms    (CUDA Measured)
    [   0   0  40  81 113 136 151 185 201 246 258 286 326 ...   0   0 ]
    passed
==== work-efficient scan, power-of-two ====
   elapsed time: 56.3608ms    (CUDA Measured)
    passed
==== work-efficient scan, non-power-of-two ====
   elapsed time: 56.3495ms    (CUDA Measured)
    passed
==== work-efficient scan with shared memory, NPOT ====
   elapsed time: 14.1294ms    (CUDA Measured)
    passed
==== thrust scan, power-of-two ====
   elapsed time: 6.22323ms    (CUDA Measured)
    passed
==== thrust scan, non-power-of-two ====
   elapsed time: 6.1984ms    (CUDA Measured)
    passed

*****************************
** STREAM COMPACTION TESTS **
*****************************
    [   3   2   1   0   2   2   0   1   0   1   2   2   1 ...   1   0 ]
==== cpu compact without scan, power-of-two ====
   elapsed time: 477.227ms    (std::chrono Measured)
    [   3   2   1   2   2   1   1   2   2   1   1   1   1 ...   1   1 ]
    passed
==== cpu compact without scan, non-power-of-two ====
   elapsed time: 471.62ms    (std::chrono Measured)
    [   3   2   1   2   2   1   1   2   2   1   1   1   1 ...   1   1 ]
    passed
==== cpu compact with scan, NPOT ====
   elapsed time: 1253.5ms    (std::chrono Measured)
    [   3   2   1   2   2   1   1   2   2   1   1   1   1 ...   1   1 ]
    passed
==== work-efficient compact, power-of-two ====
   elapsed time: 69.9303ms    (CUDA Measured)
    passed
==== work-efficient compact, non-power-of-two ====
   elapsed time: 69.2272ms    (CUDA Measured)
    passed
==== work-efficient compact with shared memory, NPOT ====
   elapsed time: 37.3776ms    (CUDA Measured)
    passed

*****************************
** RADIX SORT TESTS **
*****************************
    [ 13199 476 28810 3884 19701 31098 27003 825 22893 2677 17059 17850 20798 ... 23812   0 ]
==== cpu std::sort, NPOT ====
   elapsed time: 12501ms    (std::chrono)
    passed
==== gpu radix sort, NPOT ====
   elapsed time: 2155.5ms    (CUDA Measured)
    passed
==== gpu radix sort with shared memory, NPOT ====
   elapsed time: 1163.54ms    (CUDA Measured)
    passed
```

