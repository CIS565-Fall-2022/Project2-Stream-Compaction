University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2 - CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Di Lu
  * (TODO) [LinkedIn](), [personal website](), [twitter](), etc.
* Tested on: Windows 11, i7-12700H @ 2.30GHz 32GB, NVIDIA GeForce RTX 3050 Ti

## Introduction

In this project, I implemented the following algorithms on the GPU and tested them:

1. Exclusive Scan (CPU, Naive Scan, Work-Efficient Scan, Thrust Scan) - given an array A, output another array B such that each element b\[i\]
is a sum of a\[0\] + ... + a\[i - 1\] excluding itself.
2. Stream Compaction - given an array A, output another array B that only contains elements from A which match a criteria.

## Implementation and Results
#### CPU: Sequential Scan 
#### CPU: Stream Compact without Scan
#### CPU: Stream Compact with Scan
#### GPU: Naive GPU Scan
#### GPU: Work-Efficient GPU Scan
#### GPU: Thrust Scan
#### GPU: Stream Compaction

## Performance Analysis

* Roughly optimize the block sizes of each of your implementations for minimal
  run time on your GPU.
  * (You shouldn't compare unoptimized implementations to each other!)
  ![](img/blocksize.png)

* Compare all of these GPU Scan implementations (Naive, Work-Efficient, and
  Thrust) to the serial CPU version of Scan. Plot a graph of the comparison
  (with array size on the independent axis).
  * We wrapped up both CPU and GPU timing functions as a performance timer class for you to conveniently measure the time cost.
    * We use `std::chrono` to provide CPU high-precision timing and CUDA event to measure the CUDA performance.
    * For CPU, put your CPU code between `timer().startCpuTimer()` and `timer().endCpuTimer()`.
    * For GPU, put your CUDA code between `timer().startGpuTimer()` and `timer().endGpuTimer()`. Be sure **not** to include any *initial/final* memory operations (`cudaMalloc`, `cudaMemcpy`) in your performance measurements, for comparability.
    * Don't mix up `CpuTimer` and `GpuTimer`.
  * To guess at what might be happening inside the Thrust implementation (e.g.
    allocation, memory copy), take a look at the Nsight timeline for its
    execution. Your analysis here doesn't have to be detailed, since you aren't
    even looking at the code for the implementation.

* Write a brief explanation of the phenomena you see here.
  * Can you find the performance bottlenecks? Is it memory I/O? Computation? Is
    it different for each implementation?

* Paste the output of the test program into a triple-backtick block in your
  README.
  * If you add your own tests (e.g. for radix sort or to test additional corner
    cases), be sure to mention it explicitly.

These questions should help guide you in performance analysis on future
assignments, as well.

## Extra Credit?






