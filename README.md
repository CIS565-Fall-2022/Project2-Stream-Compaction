**University of Pennsylvania, CIS 565: GPU Programming and Architecture**

- Alex Fu
  - [LinkedIn](https://www.linkedin.com/in/alex-fu-b47b67238/)
  - [Twitter](https://twitter.com/AlexFu8304)
  - [Personal Website](https://thecger.com/)
- Tested on: Windows 10, i7-10750H @ 2.60GHz, 16GB, GTX 3060 6GB

CUDA Stream Compaction
======================

## Features Implemented

* CPU Scan and Compaction

* Naive GPU Scan

* Work-efficient GPU Scan

* GPU Steam Compaction

* Radix Sort (Extra Credit)

### Example Output

`SIZE` is $2^{20}$. The test size for non-power-of-two case is `SIZE - 3`. CUDA block size is 128.

```
****************
** SCAN TESTS **
****************
    [  41  17  34   0  19  24  28   8  12  14   5  45  31 ...  20   0 ]
==== cpu scan, power-of-two ====
   elapsed time: 0.5527ms    (std::chrono Measured)
    [   0  41  58  92  92 111 135 163 171 183 197 202 247 ... 25683436 25683456 ]
==== cpu scan, non-power-of-two ====
   elapsed time: 0.5991ms    (std::chrono Measured)
    [   0  41  58  92  92 111 135 163 171 183 197 202 247 ... 25683337 25683375 ]
    passed
==== naive scan, power-of-two ====
   elapsed time: 0.607232ms    (CUDA Measured)
    passed
==== naive scan, non-power-of-two ====
   elapsed time: 0.884736ms    (CUDA Measured)
    passed
==== work-efficient scan, power-of-two ====
   elapsed time: 0.359712ms    (CUDA Measured)
    passed
==== work-efficient scan, non-power-of-two ====
   elapsed time: 0.351232ms    (CUDA Measured)
    passed
==== thrust scan, power-of-two ====
   elapsed time: 0.16576ms    (CUDA Measured)
    passed
==== thrust scan, non-power-of-two ====
   elapsed time: 0.390144ms    (CUDA Measured)
    passed

*****************************
** STREAM COMPACTION TESTS **
*****************************
    [   1   3   2   0   1   0   2   2   2   0   1   1   1 ...   0   0 ]
==== cpu compact without scan, power-of-two ====
   elapsed time: 2.5172ms    (std::chrono Measured)
    [   1   3   2   1   2   2   2   1   1   1   3   1   3 ...   2   3 ]
    passed
==== cpu compact without scan, non-power-of-two ====
   elapsed time: 2.2528ms    (std::chrono Measured)
    [   1   3   2   1   2   2   2   1   1   1   3   1   3 ...   2   2 ]
    passed
==== cpu compact with scan, power-of-two ====
   elapsed time: 4.2481ms    (std::chrono Measured)
    passed
==== cpu compact with scan, non-power-of-two ====
   elapsed time: 4.5622ms    (std::chrono Measured)
    passed
==== work-efficient compact, power-of-two ====
   elapsed time: 0.42496ms    (CUDA Measured)
    passed
==== work-efficient compact, non-power-of-two ====
   elapsed time: 0.42096ms    (CUDA Measured)
    passed

**********************
** RADIX SORT TESTS **
**********************
    [  41  17  34   0  19  24  28   8  12  14   5  45  31 ...  20   0 ]
==== thrust sort, power-of-two ====
   elapsed time: 0.421664ms    (CUDA Measured)
    [   0   0   0   0   0   0   0   0   0   0   0   0   0 ...  49  49 ]
==== thrust sort, non-power-of-two ====
   elapsed time: 0.342016ms    (CUDA Measured)
    [   0   0   0   0   0   0   0   0   0   0   0   0   0 ...  49  49 ]
==== radix sort, power-of-two ====
   elapsed time: 20.2885ms    (CUDA Measured)
    passed
==== radix sort, non-power-of-two ====
   elapsed time: 23.1066ms    (CUDA Measured)
    passed
```

## Performance Analysis

### Performance Impacted by Array Size

CUDA block size is 128.

#### Scan

![Scan Time Impacted by Array Size (Power of Two)](img/Scan Time Impacted by Array Size (Power of Two).png)

![Scan Time Impacted by Array Size (Non Power of Two)](img/Scan Time Impacted by Array Size (Non Power of Two).png)

#### Compaction

![Compaction Time Impacted by Array Size (Power of Two)](img/Compaction Time Impacted by Array Size (Power of Two).png)

![Compaction Time Impacted by Array Size (Non Power of Two)](img/Compaction Time Impacted by Array Size (Non Power of Two).png)

#### Radix Sort

![Sort Time Impacted by Array Size (Power of Two)](img/Sort Time Impacted by Array Size (Power of Two).png)

![Sort Time Impacted by Array Size (Non Power of Two)](img/Sort Time Impacted by Array Size (Non Power of Two).png)

## Answers to the Questions

## Some Thoughts
