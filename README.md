CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* RHUTA JOSHI
  * [LinkedIn](https://www.linkedin.com/in/rcj9719/)
  * [Website](https://sites.google.com/view/rhuta-joshi)

* Tested on: Windows 10 Home, i5-7200U CPU @ 2.50GHz, NVIDIA GTX 940MX 4096 MB (Personal Laptop), RTX not supported
* GPU Compatibility: 5.0

### Introduction

Stream compaction is an important parallel computing primitive that generates a compact output buffer with selected elements of an input buffer based on some condition. Basically, given an array of elements, we want to create a new array with elements that meet a certain criteria while preserving order.
The important steps in a stream compaction algorithm are as follows:

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


In this project, I implemented stream compaction on CPU and GPU using parallel all-prefix-sum (commonly known as scan) with CUDA and analyzed the performance of each of them. The scan can be performed in two ways:

1. Naive scan
2. Work-efficient scan
    - Step 1: Upsweep scan (Parallel Reduction)

        ![](img/upsweep.jpg)

    - Step 2: Downsweep scan (Collecting scanned results) - At each level
        - Left child: Copy the parent value
        - Right child: Add the parent value and left child value copying  root value.

        [](img/downsweep.jpg)




















### References

