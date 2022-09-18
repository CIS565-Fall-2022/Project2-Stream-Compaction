CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Dongying Liu
  * [LinkedIn](https://www.linkedin.com/in/dongying-liu/), [personal website](https://vivienliu1998.wixsite.com/portfolio)
* Tested on:  Windows 11, i7-11700 @ 2.50GHz, NVIDIA GeForce RTX 3060

# Project Description
This project is the implementation of GPU stream compaction in CUDA, which will be used in the later path tracer project.
Stream compaction inculdes generally three steps:
1) Boolean Mapping: mapping the input date to a boolean array, where 1 if corresponding element meets criteria, 0 if element does not meet criteria.
2) Exclusive Scamn: run exclusive scan on the boolean array and save the result to a temporary array.
3) Scatter: According to the boolean array, if the element is 1, use the corresponding result from the scan temporary array as index, write the input data to the final result array.  
<img src="/img/stream_compaction.jpg"  width="300">

In order to implement the GPU stream compaction, the following algorithem is implemented in this project:

### CPU scan (exclusive prefix sum)
 For performance comparison and better understanding before moving to GPU.
### CPU Stream Compaction without scan and CPU Stream Compaction with scan
 For performance comparison and better understanding before moving to GPU.
### Naive GPU Scan (exclusive prefix sum)
 As the example shows, because each GPU thread might read and write to the same index of array, two buffer is needed for this naive algorithem. One is for read only and one is for write only, switch the buffer is required after every iteration.  
 <img src="/img/naive_gpu_scan.jpg"  width="300">
### Work-Efficient GPU Scan
 This algothrithm is based on the one presented by Blelloch. As the example shows, the algorithem consists of two phases: up-sweep and down-sweep.  
 The up-sweep is also know as a parallel reduction, because after this phase, the last node in the array holds the sum of all nodes in the array.  
 In the down-sweep phase, we use partial sum from the up-sweep phase to build the scan of the array. Start by replacing the last node to zero, and for each step, each node at the current level pass its own value to its left child, its right child holds the sum of itself and the former left child.  
 <img src="/img/efficient_gpu_scan.jpg"  width="300">
 
# Performance Analysis
For the performance analysis, I used blocksize of 256. I was testing on four array size, which are 2^12, 2^16, 2^20, 2^24. The less time consuming the faster the program run, the better the performance is. The time for GPU does not record the cudaMalloc and cudaMemcpy time.

## Scan (exclusive prefix sum)
<img src="/img/scan_analysis_large.png"  width="500">  <img src="/img/scan_analysis.gif"  width="500">
* As the left graph shows, when array size is from 2^8 - 2^16, CPU actually runs faster than GPU naive and work-efficient. I think the reason is for a certain number of data, the time cost on GPU for reading data from global memory can not leverage the time saved by the parallel calculation. So, the performance is quit the same or not as good as CPU.  
  However, when the amount of data reached a point, as the right graph shows, for my test is 2^20, GPU efficient algorithem starts to act way better than CPU and GPU naive algothrithm.

* No matter how large the array size is, the performance for thrust is always the best, even much better than GPU efficient. I think the reason is, for my version of GPU efficient algorithm, some thread is not used in the later iteration, but they still need to wait for the other active thread in the same warp to finish their work to become available for other work again.

* For the two GPU version, naive and work-efficient, naive cost more time, sometime even as same as CPU version. I think the most important reason is because we are ping-pong from two buffer every iteration, the cudaMemcpy cost most of the time and is very time consuming.

## Stream Compacton
<img src="/img/stream_compaction_large.png"  width="500"> <img src="/img/stream_compaction_analysis.gif"  width="500">
* Same with Scan, as the left graph shows, when array size is from 2^8 - 2^16, the two CPU functions run faster than GPU. And the reason is same as above. 
  Same, when the amount of data reached a point, as the right graph shows, for my test is 2^20, GPU efficient algorithm starts to act way better than the two CPU functions.
 * What interesting here is, the CPU without scan perfored much better than CPU with scan. So apparently, when not running parallelly, the two extra buffer created when using scan in stream compaction are very time consuming. It only acts efficiently when running parallely.

## Output of the Test Program
```
****************
** SCAN TESTS **
****************
    [  17   0   9  24   3  46  31  40   8  44  48   9  24 ...  49   0 ]
==== cpu scan, power-of-two ====
   elapsed time: 25.2266ms    (std::chrono Measured)
    [   0  17  17  26  50  53  99 130 170 178 222 270 279 ... 410923873 410923922 ]
==== cpu scan, non-power-of-two ====
   elapsed time: 24.2735ms    (std::chrono Measured)
    [   0  17  17  26  50  53  99 130 170 178 222 270 279 ... 410923806 410923816 ]
    passed
==== naive scan, power-of-two ====
   elapsed time: 21.4419ms    (CUDA Measured)
    passed
==== naive scan, non-power-of-two ====
   elapsed time: 21.3356ms    (CUDA Measured)
    passed
==== work-efficient scan, power-of-two ====
   elapsed time: 9.57216ms    (CUDA Measured)
    passed
==== work-efficient scan, non-power-of-two ====
   elapsed time: 9.46726ms    (CUDA Measured)
    passed
==== thrust scan, power-of-two ====
   elapsed time: 0.989184ms    (CUDA Measured)
    passed
==== thrust scan, non-power-of-two ====
   elapsed time: 1.25235ms    (CUDA Measured)
    passed

*****************************
** STREAM COMPACTION TESTS **
*****************************
    [   3   2   1   1   2   1   0   0   3   0   2   3   3 ...   0   0 ]
==== cpu compact without scan, power-of-two ====
   elapsed time: 36.6462ms    (std::chrono Measured)
    [   3   2   1   1   2   1   3   2   3   3   3   2   3 ...   2   3 ]
    passed
==== cpu compact without scan, non-power-of-two ====
   elapsed time: 35.2236ms    (std::chrono Measured)
    [   3   2   1   1   2   1   3   2   3   3   3   2   3 ...   1   2 ]
    passed
==== cpu compact with scan ====
   elapsed time: 94.3538ms    (std::chrono Measured)
    [   3   2   1   1   2   1   3   2   3   3   3   2   3 ...   2   3 ]
    passed
==== work-efficient compact, power-of-two ====
   elapsed time: 10.7924ms    (CUDA Measured)
    passed
==== work-efficient compact, non-power-of-two ====
   elapsed time: 11.4318ms    (CUDA Measured)
    passed
```

## Bloopers
This is more like a question than blooper.  
I used pow() in kernel function at the first time and pass two int to it. When array size reached 2^11, the algorithm began to act weiredly. It took me an hour to find out that the problem was caused by pow() I was using.  
It looks like in kernel, pow is defined as 'double result = pow(double x, double y)'  
And there is this powf() which is defined as 'float result = powf(float x, float y)'  
So, I changed it to powf(), however, it again began to act weiredly when array size reaced 2^16, I changed all the power function to bitwise eventually.  
However, is there any criteria we need to follow when using power on kernel?

