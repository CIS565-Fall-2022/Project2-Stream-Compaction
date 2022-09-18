CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Nick Moon
  * [LinkedIn](https://www.linkedin.com/in/nick-moon1/), [personal website](https://nicholasmoon.github.io/)
* Tested on: Windows 10, AMD Ryzen 9 5900HS @ 3.0GHz 32GB, NVIDIA RTX 3060 Laptop 6GB (Personal Laptop)

**This is a Stream Compaction algorithm implementation in C++ using CUDA for GPU acceleration. This allows for
compacting arrays with millions of elements in parallel on the GPU.**

### Results

*Testbench functional and performance results for scan and stream compaction on input array size of 2^25 (33554432):*
```
number of elements in input array for power of 2 tests = 33554432
number of elements in input array for non-power of 2 tests = 33554429

****************
** SCAN TESTS **
****************
    [  34  41  19  15   6   8   5  37  47  15   4  40  20 ...  44  16   0 ]
==== cpu scan, power-of-two ====
   elapsed time: 19.1805ms    (std::chrono Measured)
    [   0  34  75  94 109 115 123 128 165 212 227 231 271 ... 821815873 821815917 821815933 ]
==== cpu scan, non-power-of-two ====
   elapsed time: 19.9557ms    (std::chrono Measured)
    [   0  34  75  94 109 115 123 128 165 212 227 231 271 ... 821815825 821815842 821815853 ]
    passed
==== naive scan, power-of-two ====
   elapsed time: 29.1328ms    (CUDA Measured)
    [   0  34  75  94 109 115 123 128 165 212 227 231 271 ... 821815873 821815917 821815933 ]
    passed
==== naive scan, non-power-of-two ====
   elapsed time: 29.0673ms    (CUDA Measured)
    [   0  34  75  94 109 115 123 128 165 212 227 231 271 ... 821815825 821815842 821815853 ]
    passed
==== work-efficient scan, power-of-two ====
   elapsed time: 24.446ms    (CUDA Measured)
    [   0  34  75  94 109 115 123 128 165 212 227 231 271 ... 821815873 821815917 821815933 ]
    passed
==== work-efficient scan, non-power-of-two ====
   elapsed time: 21.2756ms    (CUDA Measured)
    [   0  34  75  94 109 115 123 128 165 212 227 231 271 ... 821815825 821815842 821815853 ]
    passed
==== thrust scan, power-of-two ====
   elapsed time: 1.16003ms    (CUDA Measured)
    [   0  34  75  94 109 115 123 128 165 212 227 231 271 ... 821815873 821815917 821815933 ]
    passed
==== thrust scan, non-power-of-two ====
   elapsed time: 1.17952ms    (CUDA Measured)
    [   0  34  75  94 109 115 123 128 165 212 227 231 271 ... 821815825 821815842 821815853 ]
    passed

*****************************
** STREAM COMPACTION TESTS **
*****************************
    [   2   3   3   0   0   1   1   0   3   1   0   1   1 ...   2   0   0 ]
==== cpu compact without scan, power-of-two ====
   elapsed time: 51.9551ms    (std::chrono Measured)
    [   2   3   3   1   1   3   1   1   1   3   2   2   2 ...   2   3   2 ]
    passed
==== cpu compact without scan, non-power-of-two ====
   elapsed time: 51.9623ms    (std::chrono Measured)
    [   2   3   3   1   1   3   1   1   1   3   2   2   2 ...   3   2   3 ]
    passed
==== cpu compact with scan ====
   elapsed time: 103.018ms    (std::chrono Measured)
    [   2   3   3   1   1   3   1   1   1   3   2   2   2 ...   2   3   2 ]
    passed
==== work-efficient compact, power-of-two ====
   elapsed time: 25.3891ms    (CUDA Measured)
    [   2   3   3   1   1   3   1   1   1   3   2   2   2 ...   2   3   2 ]
    passed
==== work-efficient compact, non-power-of-two ====
   elapsed time: 25.2508ms    (CUDA Measured)
    [   2   3   3   1   1   3   1   1   1   3   2   2   2 ...   3   2   3 ]
    passed
```


## Implementation

### Array Scan

The array scan algorithm is an algorithm that, given an input array of ints ```idata``` of size n, 
will output an an array ```odata``` with size also n, where the value at each 
index ```i``` in ```odata``` is the sum of all previous ```i-1``` elements from ```idata```.
Exclusive scan is the version implemented in this repository, as opposed to Inclusive. This means 
setting ```odata[0] = 0``` and each element in ```odata``` not including the addition of 
its correpsonding value in ```idata``` when computing the value. Below each of the different
implementations of the exclusive scan algorithm are talked in more detail.

#### CPU Implementation

The CPU implementation implements the array scan algorithm in a serial format, according to the pseudocode

```
function scan_cpu(input_array, output_array, number_of_elements):

    output_array[0] = 0
    for i in range [1, number_of_elements):
        output_array[i] = output_array[i - 1] + input_array[i - 1]

end
```

#### Naive CUDA Implemenation

The Naive CUDA implementation maps the original cpu scan approach and maps it to parallel GPU hardware.
This necessitates the use of double-buffering in order to avoid race conditions. This means swapping the
two gpu arrays that are used as input and output each iteration.

```
function scan_gpu(input_array, output_array, number_of_elements):

    let o_array_gpu_0 = input_array (this is an array on the GPU)
    let o_array_gpu_1 = o_array_gpu_0 (this is an array on the GPU)

    for all k in parallel:
        shift_array_right(k, o_array_gpu_1, o_array_gpu_0)

    for d in range [1, ceil( log_2(n) ) ]:
        for all k in parallel:
            naive_scan_iteration(k, o_array_gpu_0, o_array_gpu_1, 2^d)

end

kernel shift_array_right(thread_ID, input_array, output_array):
    if thread_ID_ == 0 then:
        output_array[0] = 0
    else:
        output_array[thread_ID] = input_array[thread_ID - 1]

end

kernel naive_scan_iteration(thread_ID, input_array, output_array, offset):
    if (thread_ID < offset) then:
        output_array[thread_ID] = input_array[thread_ID]
    else:
        output_array[thread_ID] = input_array[thread_ID - offset] + input_array[thread_ID]
    
end
```

#### Work-Efficient CUDA Implementation

To make the parallel approach more efficient, a different scheme is used. A problem with the previous method is
that 

```
function scan_gpu(input_array, output_array, number_of_elements):

    let o_array_gpu_0 = input_array (this is an array on the GPU)
    let o_array_gpu_1 = o_array_gpu_0 (this is an array on the GPU)

    for all k in parallel:
        shift_array_right(k, o_array_gpu_1, o_array_gpu_0)

    for d in range [1, ceil( log_2(n) ) ]:
        for all k in parallel:
            naive_scan_iteration(k, o_array_gpu_0, o_array_gpu_1, 2^d)

end

kernel shift_array_right(thread_ID, input_array, output_array):
    if thread_ID_ == 0 then:
        output_array[0] = 0
    else:
        output_array[thread_ID] = input_array[thread_ID - 1]

end

kernel naive_scan_iteration(thread_ID, input_array, output_array, offset):
    if (thread_ID < offset) then:
        output_array[thread_ID] = input_array[thread_ID]
    else:
        output_array[thread_ID] = input_array[thread_ID - offset] + input_array[thread_ID]
    
end
```

#### Thrust Implementation

For the thrust implementation, the input and output arrays are simply converted to thrust library
device vectors and the thrust "exclusive_scan()" function is called with them. The thrust device vector
for the output data is then transferred back to the host array.

### Stream Compaction

Stream compaction is an array algorithm that, given an input array of ints ```idata``` of size ```n```,
returns an output array ```odata``` of some size ```[0,n]``` such that ```odata``` contains only the
values ```x``` in ```idata``` that satisfy some criteria function ```f(x)```. This is essentially 
used to compact an array into a smaller size by getting rid of unneeded elements as determined by 
the criteria function ```f(x)```. Values for which ```f(x)``` return ```true``` are kept, while
values for which ```f(x)``` return false are removed.

#### CPU Implementation

#### Extra Credit

###### Not implemented :(

## Testing Strategy

The first step in the testing strategy was to figure out the optimal block size for the 
different GPU implementations of the scan and stream compaction algorithms. Data from each
implementation was collected with a constant input array size of 2^25 (33,554,432) for
powers of two block sizes from 32 to 1024, and the resutls are shown in Figure XXX below.

![](images/figures/graph_blocksize.png)
*Figure XXX: Effect of CUDA block size on runtime of scan and stream compaction.*

## Performance Analysis

As can be seen by Figure XXX below, the runtime of the scan algorithm increases linearly for each of the
different implementations, but the slope of this increase is different for each one.

![](images/figures/graph_scan.png)
*Figure 5: Effect of input array size on runtiem of scan algorithm.*

Likewise, Figure XXX below shows the runtime of the stream compaction algorithm also increases linearly
for each of the implementations, and again the slope of this increase is the only thing that changes.
However, unlike with scan, here the GPU stream compaction is significantly faster than either of the
CPU implementations for arrays with very large amounts of elements (>2000000).

![](images/figures/graph_compact.png)
*Figure 6: Effect of cuda kernel block size on average fps for uniform grid-based simulation*

As a final point of analysis, Figure 7 below displays the effect of changing the grid cell resolution with respect the
maximum neighbor search distance each boid uses to get nearby boids that influence its velocity. As can be shown,
balancing this ratio is important for improving and maintain efficiency, and the benefits of the grid-based system.
Too low a resolution, such as the first data point on the graph, and the performance is suboptimal as a lot of potentially
empty space is being encompassed by large cells. Too high a resolution, and the benefit of the grid-based system is lost.
There are now so many neighboring cells needed to be checked by a single boid, that even the increased potential
prevalence of empty cells is lost. The graph shows that having a resolution approximately equal to the maximum search distance
is optimal.

![](images/figures/ratio_vs_fps.png)
*Figure 7: Effect of the ratio of grid cell width to boid neighbor max search distance on average FPS*

### Bloopers

![](images/bloopers/blooper_graydeath.PNG)
*Blooper caused by accessing wrong index values for boid position and velocity arrays*

**For more bloopers, see images/bloopers.**

