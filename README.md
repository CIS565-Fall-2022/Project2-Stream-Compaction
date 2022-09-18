CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Ryan Tong
  * [LinkedIn](https://www.linkedin.com/in/ryanctong/), [personal website](), [twitter](), etc.
* Tested on: Windows 10, i7-8750H @ 2.20GHz 16GB, GeForce GTX 1060 6144MB (Personal Laptop)

Include analysis, etc. (Remember, this is public, so don't put
anything here that you don't want to share with the world.)

### Features
This project implements the required features of CPU based scan and stream compaction; GPU based naive scan, work-efficient scan, stream compaction (using work-efficient scan); and Thrust based scan. I roughly optimized the block size to be 256 after doing some testing and reading this article: https://oneflow2020.medium.com/how-to-choose-the-grid-size-and-block-size-for-a-cuda-kernel-d1ff1f0a7f92. 

### Performance Analysis
Here are the graphs comapring the runtimes of scan implemented on the CPU, GPU, and with Thrust. Note that Thrust is removed and the range of array sizes is shrunk for the second graph for visualization purposes.

![Scan](images/scan.png)
![Scan (Better Visualization)](images/scan_small.png)

From the graphs, we can see that CPU is faster than the work-efficent GPU implementation until the array size reaches about ~1,000,000 elements. This is suprising because the theortical complexities of these algorithms are O(n), O(nlogn), O(n) for CPU, naive, and work efficent respectively. Since the GPU implementations are paralleized we would expect that they are faster than the CPU implementation. The cause of this is likely the lack of optimizations in my GPU code and frequent reads and writes to global memory which is slow. An implementation using shared memory would improve the memory access speeds. Further more, the indexing of scan is inefficent since there are many inactive threads that could be retired in a warp if they were consecutive. 

The Thrust implementations are significantly slower than both GPU and CPU implementation which is likely due to some implementation error that I was unable to solve.

We can see these inefficenies reflected again in the stream compaction run times:

![Stream Compaction](images/scan.png)
![Stream Compaction (Better Visualization)](images/scan_small.png)

### Program Output
```

****************
** SCAN TESTS **
****************
    [  29  48   7  23  28  16  45  34   2  47  35   3  16 ...  48   0 ]
==== cpu scan, power-of-two ====
   elapsed time: 2.702ms    (std::chrono Measured)
    [   0  29  77  84 107 135 151 196 230 232 279 314 317 ... 12845931 12845979 ]
==== cpu scan, non-power-of-two ====
   elapsed time: 2.7096ms    (std::chrono Measured)
    [   0  29  77  84 107 135 151 196 230 232 279 314 317 ... 12845838 12845880 ]
    passed
==== naive scan, power-of-two ====
   elapsed time: 4.85891ms    (CUDA Measured)
    passed
==== naive scan, non-power-of-two ====
   elapsed time: 4.5247ms    (CUDA Measured)
    passed
==== work-efficient scan, power-of-two ====
   elapsed time: 2.23603ms    (CUDA Measured)
    passed
==== work-efficient scan, non-power-of-two ====
   elapsed time: 2.10493ms    (CUDA Measured)
    passed
==== thrust scan, power-of-two ====
   elapsed time: 35.5277ms    (CUDA Measured)
    passed
==== thrust scan, non-power-of-two ====
   elapsed time: 27.5845ms    (CUDA Measured)
    passed

*****************************
** STREAM COMPACTION TESTS **
*****************************
    [   3   2   3   1   2   0   3   0   2   3   1   3   2 ...   2   0 ]
==== cpu compact without scan, power-of-two ====
   elapsed time: 6.4836ms    (std::chrono Measured)
    [   3   2   3   1   2   3   2   3   1   3   2   3   1 ...   1   2 ]
    passed
==== cpu compact without scan, non-power-of-two ====
   elapsed time: 5.3097ms    (std::chrono Measured)
    [   3   2   3   1   2   3   2   3   1   3   2   3   1 ...   2   1 ]
    passed
==== cpu compact with scan ====
   elapsed time: 14.7061ms    (std::chrono Measured)
    [   3   2   3   1   2   3   2   3   1   3   2   3   1 ...   1   2 ]
    passed
==== work-efficient compact, power-of-two ====
   elapsed time: 2.84058ms    (CUDA Measured)
    passed
==== work-efficient compact, non-power-of-two ====
   elapsed time: 2.50528ms    (CUDA Measured)
    passed
```
