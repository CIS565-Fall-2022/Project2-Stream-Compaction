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

### Extra credit

#### Performance
My work efficient scan halves the total number of threads launched each iteration, so the work efficient scan is faster at than naive and CPU implementation at array sizes of 2^18 and larger. This means less threads are idling and taking up space on the GPU multi-processors.

#### Radix sort

I implemented Radix sort on the CPU, wrote two test cases, and added Radix sort on the GPU, which calls my work efficient scan. The functions to look at are:
- `radixSort` in `naive.cu`
- `radixSort` in `cpu.cu`

A few notes: you can pass in the number of bits you want to sort by, which should be `ilog2ceil(MAX_ARRAY_ELEMENT_VALUE)`. Also, I assumed for simplicity that each element is a positive integer (although still using int and not unsigned int types) so I can just use a bitmask to compact the arrays. Finally, to test, the array size should not be too close to 2^31 because of integer overflow issues...  

#### Sample program output
Settings: `blockSize` = 128, `SIZE` = 1 << 20

```

****************
** SCAN TESTS **
****************
    [   8  40  25  15  22   4  39  46  44  28   2  44  33 ...  30   0 ]
==== cpu scan, power-of-two ====
   elapsed time: 1.5927ms    (std::chrono Measured)
    [   0   8  48  73  88 110 114 153 199 243 271 273 317 ... 25674057 25674087 ]
==== cpu scan, non-power-of-two ====
   elapsed time: 1.7177ms    (std::chrono Measured)
    [   0   8  48  73  88 110 114 153 199 243 271 273 317 ... 25673959 25673964 ]
    passed
==== naive scan, power-of-two ====
   elapsed time: 0.634144ms    (CUDA Measured)
    passed
==== naive scan, non-power-of-two ====
   elapsed time: 0.602528ms    (CUDA Measured)
    passed
==== work-efficient scan, power-of-two ====
   elapsed time: 0.457504ms    (CUDA Measured)
    passed
==== work-efficient scan, non-power-of-two ====
   elapsed time: 0.479328ms    (CUDA Measured)
    passed
==== thrust scan, power-of-two ====
   elapsed time: 0.340864ms    (CUDA Measured)
    passed
==== thrust scan, non-power-of-two ====
   elapsed time: 0.279648ms    (CUDA Measured)
    passed

*****************************
** STREAM COMPACTION TESTS **
*****************************
    [   2   0   2   2   3   1   2   1   1   2   2   0   3 ...   0   0 ]
==== cpu compact without scan, power-of-two ====
   elapsed time: 2.2561ms    (std::chrono Measured)
    [   2   2   2   3   1   2   1   1   2   2   3   1   2 ...   1   1 ]
    passed
==== cpu compact without scan, non-power-of-two ====
   elapsed time: 2.1692ms    (std::chrono Measured)
    [   2   2   2   3   1   2   1   1   2   2   3   1   2 ...   1   1 ]
    passed
==== cpu compact with scan ====
   elapsed time: 5.498ms    (std::chrono Measured)
    [   2   2   2   3   1   2   1   1   2   2   3   1   2 ...   1   1 ]
    passed
==== work-efficient compact, power-of-two ====
   elapsed time: 0.541312ms    (CUDA Measured)
    passed
==== work-efficient compact, non-power-of-two ====
   elapsed time: 0.490016ms    (CUDA Measured)
    passed

*****************************
** RADIX SORT TESTS **
*****************************
    [ 23914 19768 8186 18274 8083 25457 19314 28109 22549 17382 25706 27432 8575 ... 4256   0 ]
==== cpu radix sort, power-of-two ====
    [   0   0   0   0   0   0   0   0   0   0   0   0   0 ... 32767 32767 ]
==== radix sort, power of two ====
   elapsed time: 0.490016ms    (CUDA Measured)
    passed
==== cpu radix sort, non-power-of-two ====
   elapsed time: 109.43ms    (std::chrono Measured)
    [   0   0   0   0   0   0   0   0   0   0   0   0   0 ... 32767 32767 ]
==== radix sort, non-power of two ====
   elapsed time: 0.490016ms    (CUDA Measured)
    passed
```