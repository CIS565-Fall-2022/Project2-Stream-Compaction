CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

Constance Wang
  * [LinkedIn](https://www.linkedin.com/in/conswang/)

Tested on AORUS 15P XD laptop with specs:  
- Windows 11 22000.856  
- 11th Gen Intel(R) Core(TM) i7-11800H @ 2.30GHz 2.30 GHz  
- NVIDIA GeForce RTX 3070 Laptop GPU  

I implemented the following parallel algorithms on the GPU and benchmarked them against my own CPU implementations and Thrust on the GPU:  
- Naive Scan
- Work efficient scan
- Stream compaction
- Radix sort

I roughly optimized the block size for each algorithm by seeing what block size performed the fastest on arrays of size 2^22 (approx 4 million).

| Block Size |	Runtime - Naive (ms)	| Runtime - Work efficient (ms) |
| ----------- | ----------- | ----------- |
32 |	4.21734 |	1.32106
64	|2.11395	|1.36259
128	|2.09267|	1.26221
256|	2.09258	|1.28563
384	|2.11395	|1.3327
768	|2.11405|	1.26701

Performance was pretty similar for most block sizes, but started to suffer for both naive and work efficient at around 32 or 64 threads per block. In this case, I decided to use a block size of 128 threads to compare the algorithms on different array sizes.  

![](img/Performance%20of%20Scan%20Implementations%20on%20Different%20Array%20Sizes.svg)

| Array size | CPU (m/s)    | Naive (m/s)  | Work efficient (m/s) | Thrust (m/s)  |
|------------|---------|----------|----------------|----------|
| 65536      | 0.1023  | 0.158464 | 0.266592       | 0.045472 |
| 262144     | 0.399   | 0.2616   | 0.33888        | 0.194144 |
| 1048576    | 1.6835  | 0.636288 | 0.472416       | 0.351648 |
| 4194304    | 6.392   | 2.20544  | 1.27302        | 0.523776 |
| 16777216   | 25.5751 | 8.98938  | 4.05302        | 1.09213  |
| 67108864   | 100.736 | 38.8708  | 15.4414        | 2.14362  |
| 268435456  | 410.365 | 169.486  | 60.6265        | 6.23341  |

### Analysis
The CPU implementation's run time appears to be linear with respect to the number of array elements. This makes sense because each element is processed one at a time inside a for loop.  

Thrust is the fastest by far. This is probably because they are using shared memory, while all of my implementations only use global memory which is much slower to access, making each kernel thread slower. And maybe other optimizations as well.  

The work-efficient scan is faster than the naive scan. This should be because I made optimizations (see next section) to reduce the number of threads at each iteration, whereas the naive scan still launches n threads each iteration.

In all implementations, computation should not be a performance bottleneck, since each kernel runs in about O(1) time, we can't really do better than that. 

Aside from the above trends, memory IO (cudaMemcpy) is a giant performance bottleneck. This is not shown in the performance graph since we start measuring runtime after the initial cudaMemcpy and stop measuring before the final cudaMemcpy. Still, cudaMemcpy runs in O(n) time, which effectively makes any GPU algorithm O(n), even though the actual implementation of scan runs in O(log n).  

However, in practice, cudaMemcpy is still very fast, probably because the hardware bandwith for copying data from host to device is very large. For example, on an array of size 2^26, I ran my Radix sort algorithms and the CPU implementation took about 7 seconds (7131.1ms). Meanwhile the GPU implementation took about 1 second (826.585ms) including the cudaMemcpy, and half a second (434.71ms) without the cudaMemcpy. This means that while the cudaMemcpy is still a huge bottleneck on the GPU performance (taking up about half the runtime), it isn't too the point of being linear time, even at large numbers. In the future, I could try to measure the bandwidth of cudaMemcpy on my GPU.

### Extra credit

#### Performance
My work efficient scan halves the total number of threads launched each iteration, this means less threads are idling and taking up space on the GPU multi-processors while other threads could be running. As a result, the work efficient scan is faster at than naive and CPU implementation at array sizes of 2^18 and larger.

#### Radix sort

I implemented Radix sort on the CPU, wrote two test cases, and added Radix sort on the GPU, which calls my work efficient scan. The functions to look at are:
- `radixSort` in `naive.cu`
- `radixSort` in `cpu.cu`

A few notes: you can pass in the number of bits you want to sort by, which should be `ilog2ceil(MAX_ARRAY_ELEMENT_VALUE)`. Also, I assumed for simplicity that each element is a positive integer (although still using int and not unsigned int types) so I can just use a bitmask to compact the arrays. Finally, to test, the array size should not be too close to 2^31 because of integer overflow issues...  

#### Sample program output
Settings: `blockSize` = 128, `SIZE` = 1 << 26

```

****************
** SCAN TESTS **
****************
    [  29  25   4  37   8  30  21  31   8  21  22  19   3 ...  49   0 ]
==== cpu scan, power-of-two ====
   elapsed time: 102.046ms    (std::chrono Measured)
    [   0  29  54  58  95 103 133 154 185 193 214 236 255 ... 1643658734 1643658783 ]
==== cpu scan, non-power-of-two ====
   elapsed time: 99.084ms    (std::chrono Measured)
    [   0  29  54  58  95 103 133 154 185 193 214 236 255 ... 1643658648 1643658670 ]
    passed
==== naive scan, power-of-two ====
   elapsed time: 38.8846ms    (CUDA Measured)
    passed
==== naive scan, non-power-of-two ====
   elapsed time: 37.4897ms    (CUDA Measured)
    passed
==== work-efficient scan, power-of-two ====
   elapsed time: 15.4577ms    (CUDA Measured)
    passed
==== work-efficient scan, non-power-of-two ====
   elapsed time: 15.4086ms    (CUDA Measured)
    passed
==== thrust scan, power-of-two ====
   elapsed time: 2.09901ms    (CUDA Measured)
    passed
==== thrust scan, non-power-of-two ====
   elapsed time: 2.36995ms    (CUDA Measured)
    passed

*****************************
** STREAM COMPACTION TESTS **
*****************************
    [   0   3   2   3   3   1   3   2   2   3   1   3   0 ...   3   0 ]
==== cpu compact without scan, power-of-two ====
   elapsed time: 137.514ms    (std::chrono Measured)
    [   3   2   3   3   1   3   2   2   3   1   3   1   3 ...   2   3 ]
    passed
==== cpu compact without scan, non-power-of-two ====
   elapsed time: 137.567ms    (std::chrono Measured)
    [   3   2   3   3   1   3   2   2   3   1   3   1   3 ...   2   1 ]
    passed
==== cpu compact with scan ====
   elapsed time: 348.893ms    (std::chrono Measured)
    [   3   2   3   3   1   3   2   2   3   1   3   1   3 ...   2   3 ]
    passed
==== work-efficient compact, power-of-two ====
   elapsed time: 19.1836ms    (CUDA Measured)
    passed
==== work-efficient compact, non-power-of-two ====
   elapsed time: 19.1201ms    (CUDA Measured)
    passed

*****************************
** RADIX SORT TESTS **
*****************************
    [ 31399 13580 25635 22845 23360 14322 9628 3467 20074 16251 14385 30083 26014 ... 230   0 ]
==== cpu radix sort, power-of-two ====
   elapsed time: 7131.1ms    (std::chrono Measured)
    [   0   0   0   0   0   0   0   0   0   0   0   0   0 ... 32767 32767 ]
==== radix sort, power of two ====
   elapsed time: 826.585ms    (CUDA Measured)
    passed
==== cpu radix sort, non-power-of-two ====
   elapsed time: 7102.31ms    (std::chrono Measured)
    [   0   0   0   0   0   0   0   0   0   0   0   0   0 ... 32767 32767 ]
==== radix sort, non-power of two ====
   elapsed time: 788.974ms    (CUDA Measured)
    passed
```
