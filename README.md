CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Shutong Wu
  * [LinkedIn](https://www.linkedin.com/in/shutong-wu-214043172/)
  * [Email](shutong@seas.uepnn.edu)
* Tested on: Windows 10, i7-10700K CPU @ 3.80GHz, RTX3080, SM8.6, Personal Computer 

## GPU Scan and Stream Compaction
Implementation including:
- CPU Version of Scan
- CPU Version of Compact with Scan and without Scan
- GPU Naive Version
- GPU Work-Efficient Scan
- GPU Work-Effient Compact
- CUDA Thrust Library Scan and Compaction
- Radix Sort(Extra Credit)
- TODO: Shared Memory Optimization/Warp Partitioning(Index Optimization)

## Performance Analysis
- We use runtime to measure the performance of each version of scan/compaction, the less time it took to run, the better the performance.
- We will test with arrays that have sizes from 2^8 to 2^19, and compare the performances of different types of scans and compacts
- All number varies from 0 to 99, and the run time is mesasured in ms;
- NPOT in the following charts means Non Power of Two

### Performance Analysis of Scan
![Scan](./img/Scan.png)
- In the Scan process with normal size of arrays, CPU performs way better than GPU, no matter GPU is using naive scan or work efficient scan;
- When array size gets really large(more than 100K), GPU naive scan performs better than CPU(perheps because of parallelism's advantage towards massive work), and GPU work efficient scan has the second best performance, while CPU has the worst performance with a huge size of array(like 2^22)
- There is no clear difference for the same method if we use array that has a power of two size or Non-Power-Of-Two size. Because in our GPU method we pad NPOT arrays to power-of-two arrays, and in CPU few elements would not cause a obvious difference.

### Performance Analysis of Compaction
![Compact](./img/Compact.png)
- For a small size of array, GPU computation is slower than CPU, but for large size GPU run better than CPU, the best performance over all is GPU-Work Efficient Scan, and the worst is CPU-Compact With Scan(since we are including the time doing scan so this is the slowest)
- Scan for CPU in general is not a good choice because there are extra allocating work and heavy computation task for CPU when facing a large size of array; using GPU and especially work efficient scan is fast both in real task and in time complexity theoritically. It can be faster when using shared memory, which I will mention later.


### Performance Analysis of Radix Sort
![RadixSort](./img/RadixSort.png)
- Implemented using the method on GPU Gem 
- So for this one I do comparison only to see if the sorted result matched together(and they did!)
- ThrustSort only took very little time to sort and does not increase time when arraySize increases, while RadixSort's runtime increases when array becomes larger


### Why is My GPU Approach So Slow?
- The main reason why our GPU is slower than CPU is because all computation uses global memory in GPU. Getting data from global memory is costly in terms of performance.(Even though when facing large size of array parallelism become GPU's biggest advantage over CPU)
- When executing kernel function, our current method/code will leave a lot of separate data thus cost warp divergence and performance lag. The way to optimize it is to use warp partitioning and a new way of reduction(change index mostly).

### Printout Example for 2^15 Array
```
****************
** SCAN TESTS **
****************
    [  28  28  54  91  22  85  76  37  38  34  40  64  55 ...  64   0 ]
==== radix sort, power-of-two ====
   elapsed time: 2.82013ms    (CUDA Measured)
    [   0   0   0   0   0   0   0   0   0   0   0   0   0 ...  99  99 ]
==== radix sort, non-power-of-two ====
   elapsed time: 3.62918ms    (CUDA Measured)
    [   0   0   0   0   0   0   0   0   0   0   0   0   0 ...  99  99 ]
==== radix sort, thrust power of two ====
   elapsed time: 0.002048ms    (CUDA Measured)
    [   0   0   0   0   0   0   0   0   0   0   0   0   0 ...  99  99 ]
==== radix sort, thrust non power of two ====
   elapsed time: 0.001024ms    (CUDA Measured)
    [   0   0   0   0   0   0   0   0   0   0   0   0   0 ...  99  99 ]
==== cpu scan, power-of-two ====
   elapsed time: 0.0594ms    (std::chrono Measured)
    [   0  28  56 110 201 223 308 384 421 459 493 533 597 ... 1612129 1612193 ]
==== cpu scan, non-power-of-two ====
   elapsed time: 0.0575ms    (std::chrono Measured)
    [   0  28  56 110 201 223 308 384 421 459 493 533 597 ... 1612048 1612056 ]
    passed
==== naive scan, power-of-two ====
   elapsed time: 0.100352ms    (CUDA Measured)
    [   0  28  56 110 201 223 308 384 421 459 493 533 597 ... 1612129 1612193 ]
    passed
==== naive scan, non-power-of-two ====
   elapsed time: 0.099328ms    (CUDA Measured)
    passed
==== work-efficient scan, power-of-two ====
   elapsed time: 0.157696ms    (CUDA Measured)
    [   0  28  56 110 201 223 308 384 421 459 493 533 597 ... 1612129 1612193 ]
    passed
==== work-efficient scan, non-power-of-two ====
   elapsed time: 0.160768ms    (CUDA Measured)
    [   0  28  56 110 201 223 308 384 421 459 493 533 597 ... 1612048 1612056 ]
    passed
==== thrust scan, power-of-two ====
   elapsed time: 21.2378ms    (CUDA Measured)
    [   0  28  56 110 201 223 308 384 421 459 493 533 597 ... 1612129 1612193 ]
    passed
==== thrust scan, non-power-of-two ====
   elapsed time: 1.2287ms    (CUDA Measured)
    [   0  28  56 110 201 223 308 384 421 459 493 533 597 ... 1612048 1612056 ]
    passed

*****************************
** STREAM COMPACTION TESTS **
*****************************
    [   0   0   3   2   0   0   0   2   2   2   0   0   2 ...   1   0 ]
==== cpu compact without scan, power-of-two ====
   elapsed time: 0.1174ms    (std::chrono Measured)
    [   3   2   2   2   2   2   3   1   1   2   3   2   2 ...   2   1 ]
    passed
==== cpu compact without scan, non-power-of-two ====
   elapsed time: 0.1182ms    (std::chrono Measured)
    [   3   2   2   2   2   2   3   1   1   2   3   2   2 ...   2   1 ]
    passed
==== cpu compact with scan ====
   elapsed time: 0.1943ms    (std::chrono Measured)
    [   3   2   2   2   2   2   3   1   1   2   3   2   2 ...   2   1 ]
    passed
==== work-efficient compact, power-of-two ====
   elapsed time: 0.17408ms    (CUDA Measured)
    [   3   2   2   2   2   2   3   1   1   2   3   2   2 ...   2   1 ]
    passed
==== work-efficient compact, non-power-of-two ====
   elapsed time: 0.171008ms    (CUDA Measured)
    [   3   2   2   2   2   2   3   1   1   2   3   2   2 ...   2   1 ]
    passed
```
