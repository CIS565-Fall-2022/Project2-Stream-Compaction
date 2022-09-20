CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**


* Eyad Almoamen
  * [LinkedIn](https://www.linkedin.com/in/eyadalmoamen/), [personal website](https://eyadnabeel.com)
* Tested on: Windows 11, i7-10750H CPU @ 2.60GHz 2.59 GHz 16GB, RTX 2070 Super Max-Q Design 8GB (Personal Computer)

Introduction
======================
I implemented exclusive scan on the CPU and on the GPU using both the naive and work-efficient methods. I've also implemented stream compaction

Analysis
======================
**Effect of Block Size on performance**
I ran the algorithms with variation in block size on arrays of size n = 2^14 elements, and the following graph shows the results:

![](img/blocksize.png)

There doesn't seem to be any sort of conclusive relation between blocksize and performance.

**Effect of number of elements on performance**
(I ran into a bug which rendered the algorithm incapable of running on arrays larger than 2^14, and therefore was not able to produce any meaningful results especially in comparison with the cpu)