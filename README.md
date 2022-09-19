CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* RHUTA JOSHI
  * [LinkedIn](https://www.linkedin.com/in/rcj9719/)
  * [Website](https://sites.google.com/view/rhuta-joshi)

* Tested on: Windows 10 Home, i5-7200U CPU @ 2.50GHz, NVIDIA GTX 940MX 4096 MB (Personal Laptop), RTX not supported
* GPU Compatibility: 5.0

### Introduction

Stream Compaction
---
Stream compaction is an important parallel computing primitive that generates a compact output buffer with selected elements of an input buffer based on some condition. Basically, given an array of elements, we want to create a new array with elements that meet a certain criteria while preserving order.
The important steps in a parallel stream compaction algorithm are as follows:

![](img/stream-compaction.jpg)

1. Step 1: Mapping - Compute a temporary array containing
    - 1 if corresponding element meets criteria
    - 0 if element does not meet criteria
2. Step 2: Scanning - We can use one of the scanning techniques expanded below to run an exclusive scan on the mapped temporary array
    - Naive scan
    - Work-efficient scan
3. Step 3: Scattering - Insert input data at index obtained from scanned buffer if criteria is set to true
    - Result of scan is index into final array
    - Only write an element if temporary array has a 1

Parallel Scanning
---
In this project, I implemented stream compaction on CPU and GPU using parallel all-prefix-sum (commonly known as scan) with CUDA and analyzed the performance of each of them. The sequential scan algorithm is poorly suited to GPUs because it does not take advantage of the GPU's data parallelism. The parallel version of scan that utilizes the parallel processors of a GPU to speed up its computation. The parallel scan can be performed in two ways:

1. Naive scan - This is an O(nlogn) algorithm which iteratively adds elements with an offset.
2. Work-efficient scan - This is an O(n) algorithm
    - Step 1: **Upsweep scan** (Parallel Reduction phase) - In this, we traverse the tree from leaves to root computing partial sums at internal nodes of the tree. At the end of this phase, the root node (the last node in the array) holds the sum of all nodes in the array.

        ![](img/upsweep.jpg)

    - Step 2: **Downsweep scan** (Collecting scanned results) - In the down-sweep phase, we traverse back down the tree from the root, using the partial sums from the reduce phase to build the scan in place on the array. At each level,
        - Left child: Copy the parent value
        - Right child: Add the parent value and left child value copying  root value.

        ![](img/downsweep.jpg)


### Performance Analysis


### References

1. GPU Parallel Algorithms Course Presentation - CIS 5650 - Fall 2022
2. GPU Gems 3, Chapter 39 - [Parallel Prefix Sum (Scan) with CUDA](https://developer.nvidia.com/gpugems/GPUGems3/gpugems3_ch39.html)