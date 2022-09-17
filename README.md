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
 This algothrithem is based on the one presented by Blelloch. As the example shows, the algorithem consists of two phases: up-sweep and down-sweep.  
 The up-sweep is also know as a parallel reduction, because after this phase, the last node in the array holds the sum of all nodes in the array.  
 In the down-sweep phase, we use partial sum from the up-sweep phase to build the scan of the array. Start by replacing the last node to zero, and for each step, each node at the current level pass its own value to its left child, its right child holds the sum of itself and the former left child.  
 <img src="/img/efficient_gpu_scan.jpg"  width="300">
 
