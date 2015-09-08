CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* (TODO) YOUR NAME HERE
* Tested on: (TODO) Windows 22, i7-2222 @ 2.22GHz 22GB, GTX 222 222MB (Moore 2222 Lab)

### (TODO: Your README)

Include analysis, etc. (Remember, this is public, so don't put
anything here that you don't want to share with the world.)

Instructions (delete me)
========================

This is due **INSTRUCTOR TODO**.

**Summary:** In this project, you'll implement GPU stream compaction in CUDA,
from scratch. This algorithm is widely used, and will be important for
accelerating your path tracer project.

In addition to being useful for your path tracer, this project is meant to
reorient your algorithmic thinking to the way of the GPU. On GPUs, many
algorithms can benefit from massive parallelism and, in particular, data
parallelism: executing the same code many times simultaneously with different
data.

You'll implement a few different versions of the *Scan* (*Prefix Sum*)
algorithm. First, you'll implement a CPU version of the algorithm to reinforce
your understanding. Then, you'll write a few GPU implementations: "naive" and
"work-efficient." Finally, you'll use some of these to implement GPU stream
compaction.

**Algorithm overview & details:** There are two primary references for details
on the implementation of scan and stream compaction.

* The [slides on Parallel Algorithms](https://github.com/CIS565-Fall-2015/cis565-fall-2015.github.io/raw/master/lectures/2-Parallel-Algorithms.pptx)
  for Scan, Stream Compaction, and Work-Efficient Parallel Scan.
* [GPU Gems 3, Chapter 39](http://http.developer.nvidia.com/GPUGems3/gpugems3_ch39.html).

Your GPU stream compaction implementation will live inside of the
`stream_compaction` subproject. This way, you will be able to easily copy it
over for use in your GPU path tracer.


## Part 0: The Usual

This project (and all other CUDA projects in this course) requires an NVIDIA
graphics card with CUDA capability. Any card with Compute Capability 2.0
(`sm_20`) or greater will work. Check your GPU on this
[compatibility table](https://developer.nvidia.com/cuda-gpus).
If you do not have a personal machine with these specs, you may use those
computers in the Moore 100B/C which have supported GPUs.

**HOWEVER**: If you need to use the lab computer for your development, you will
not presently be able to do GPU performance profiling. This will be very
important for debugging performance bottlenecks in your program.

### Useful existing code

* `stream_compaction/common.h`
  * `checkCUDAError` macro: checks for CUDA errors and exits if there were any.
  * `ilog2ceil(x)`: computes the ceiling of log2(x), as an integer.
* `main.cpp`
  * Some testing code for your implementations.


## Part 1: CPU Scan & Stream Compaction

In `stream_compaction/cpu.cu`, implement:

* `StreamCompaction::CPU::scan`: compute an exclusive prefix sum.
* `StreamCompaction::CPU::compactWithoutScan`: stream compaction without using
  the `scan` function.
* `StreamCompaction::CPU::compactWithScan`: stream compaction using the `scan`
  function. Map the input array to an array of 0s and 1s, scan it, and use
  Scatter to produce the output. You will need a **CPU** Scatter implementation
  for this (see slides or GPU Gems chapter for an explanation).

These implementations should only be a few lines long.


## Part 2: Naive GPU Scan Algorithm

### 2.1. Global Memory Scan

In `stream_compaction/naive.cu`, implement:

* `StreamCompaction::Naive::scan`

This uses the "Naive" algorithm from GPU Gems 3, Section 39.2.1. However, note
that they use shared memory in Example 39-1; don't do that yet. Instead, write
this using global memory only. As a result of this, you will have to do
`ilog2ceil(n)` separate kernel invocations.

Make sure your implementation works on non-power-of-two sized arrays (see
`ilog2ceil`).


## Part 3: Work-Efficient GPU Scan & Stream Compaction

In `stream_compaction/efficient.cu`, implement:

* `StreamCompaction::Efficient::scan`
* `StreamCompaction::Efficient::compact`

This is equivalent to the "Work-Efficient Parallel Scan" from the slides and
*GPU Gems 3* section 39.2.2. You will need to implement the Scatter algorithm
presented in the slides and the GPU Gems chapter.

In `stream_compaction/common.cu`, implement these for use in `compact`:

* `StreamCompaction::Common::kernMapToBoolean`
* `StreamCompaction::Common::kernScatter`

Beware of errors in Example 39-2 in the book; the pseudocode (Examples 3/4) is
correct, but the CUDA code has a few errors (missing braces, bad indentation,
etc.)

Make sure your implementation works on non-power-of-two sized arrays (see
`ilog2ceil`).


## Part 4: Using Thrust's Implementation

In `stream_compaction/thrust.cu`, implement:

* `StreamCompaction::Thrust::scan`

This should be a very short function which wraps a call to the Thrust library
function `thrust::exclusive_scan(first, last, result)`.

To measure timing, be sure to exclude memory operations by passing
`exclusive_scan` a `thrust::device_vector` (which is already allocated on the
GPU).  You can create a `thrust::device_vector` by creating a
`thrust::host_vector` from the given pointer, then casting it.


## Part 5: Radix Sort (Extra Credit) (+10)

Add an additional module to the `stream_compaction` subproject. Implement radix
sort using one of your scan implementations.


## Write-up

1. Update all of the TODOs at the top of this README.
2. Add a description of this project including a list of its features.
3. Add your performance analysis (see below).

All extra credit features must be documented in your README, explaining its
value (with performance comparison, if applicable!) and showing an example how
it works. For radix sort, show how it is called and an example of its output.

Always profile with Release mode builds and run without debugging.

### Questions

* Roughly optimize the block sizes of each of your implementations for minimal
  run time on your GPU.
  * (You shouldn't compare unoptimized implementations to each other!)

* Compare all of these GPU Scan implementations (Naive, Work-Efficient, and
  Thrust) to the serial CPU version of Scan. Plot a graph of the comparison
  (with array size on the independent axis).
  * You should use CUDA events for timing. Be sure **not** to include any
    explicit memory operations in your performance measurements, for
    comparability.
  * To guess at what might be happening inside the Thrust implementation, take
    a look at the Nsight timeline for its execution.

* Write a brief explanation of the phenomena you see here.
  * Can you find the performance bottlenecks? Is it memory I/O? Computation? Is
    it different for each implementation?

These questions should help guide you in performance analysis on future
assignments, as well.

## Submit

If you have modified any of the `CMakeLists.txt` files at all (aside from the
list of `SOURCE_FILES`), you must test that your project can build in Moore
100B/C. Beware of any build issues discussed on the Google Group.

1. Open a GitHub pull request so that we can see that you have finished.
   The title should be "Submission: YOUR NAME".
2. Send an email to the TA (gmail: kainino1+cis565@) with:
   * **Subject**: in the form of `[CIS565] Project 0: PENNKEY`
   * Direct link to your pull request on GitHub
   * In the form of a grade (0-100+) with comments, evaluate your own
     performance on the project.
   * Feedback on the project itself, if any.
