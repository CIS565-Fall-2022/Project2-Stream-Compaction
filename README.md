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

  - [9] CPU radix sort (Part 6)

  - [A] GPU radix sort using [3] work-efficient scan (Part 6)

  - [B] GPU radix sort using [4] optimized work-efficient scan (Part 7)

### Brief Overview of Extra-Credit Implementations

#### [3] Work-Efficient Scan Not Slower than CPU

I knew that after each iteration of partial upsweep/downsweep the number of active threads halves and keeping them compact would significantly reduce control divergence, so I optimized it at the very beginning. Then it was faster than CPU.

#### [4] Work-Efficient Scan Optimized with Shared Memory

To utilize shared memory, we need to consider blockwisely. As shown in the picture [[GPU Gems 3 Figure 39-6](https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda)], the idea can be divided into three steps:

![](https://developer.nvidia.com/sites/all/modules/custom/gpugems/books/GPUGems3/elementLinks/39fig06.jpg)

- First, perform scan on each block and record the sum of each block into an auxiliary array
- Then, scan the auxiliary array to get offset to be added to each block
- Last, add offset back to each block

In the second step, however, the size of auxiliary array isn't always less than block size when 

#### [A, B] Radix Sort

##### Implementation

Basically, my implementation followed the idea shown by this picture [[GPU Gems 3 Figure 39-14](https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda)]

![](https://developer.nvidia.com/sites/all/modules/custom/gpugems/books/GPUGems3/elementLinks/39fig14.jpg)



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

For GPU naive scan and work-efficient scan without shared memory usage, I actually didn't use a fixed block size. Instead, I let the block size change depending on **data amount**, **number of SMs** and **max block size**. Here is the code (in `stream_compaction/common.h`):

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

What makes it more interesting is that when I used NSight to peek how `thrust::exclusive_scan` worked, it also launched the kernel `DeviceScanKernel` with block size of 128.

### Comparison of Different Implementations (to answer Q2)

#### Scan



#### Stream Compaction



#### Radix sort

To get a CPU baseline to compare my GPU radix sort with, I first implemented a simple CPU radix sort function `radixsort ` in [`stream_compaction/cpu.h`](./stream_compaction/cpu.h) and [`stream_compaction/cpu.cu`](./stream_compaction/cpu.cu). I also tried `std::sort`, which turned out to be equally fast as my CPU radix sort.









### Bottlenecks of Different Implementations (to answer Q3)



#### Naive Scan

#### Work-Efficient Scan

#### Work-Efficient Scan with Shared Memory

#### 



### Test Program Output Example (to answer Q4)

I added additional tests for [4, 8, A and B]. This example was tested with `SIZE = 1 << 28`.

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
==== cpu radix sort, NPOT ====
   elapsed time: 12408.4ms    (std::chrono)
    passed
==== gpu radix sort, NPOT ====
   elapsed time: 2155.5ms    (CUDA Measured)
    passed
==== gpu radix sort with shared memory, NPOT ====
   elapsed time: 1163.54ms    (CUDA Measured)
    passed
```

