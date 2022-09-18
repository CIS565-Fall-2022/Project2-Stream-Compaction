CUDA Stream Compaction
======================

### University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2 ###

* (TODO) YOUR NAME HERE
  * (TODO) [LinkedIn](), [personal website](), [twitter](), etc.
* Tested on: (TODO) Windows 22, i7-2222 @ 2.22GHz 22GB, GTX 222 222MB (Moore 2222 Lab)

### (TODO: Your README)

Include analysis, etc. (Remember, this is public, so don't put
anything here that you don't want to share with the world.)



### Sample Output for Scan and Stream Compaction Test

```
****************
** SCAN TESTS **
****************
    [  24  35  22  36   2  22   7  32  30  28   9  42  48 ...  26   0 ]
==== cpu scan, power-of-two ====
   elapsed time: 0.0017ms    (std::chrono Measured)
    [   0  24  59  81 117 119 141 148 180 210 238 247 289 ... 5969 5995 ]
==== cpu scan, non-power-of-two ====
   elapsed time: 0.0011ms    (std::chrono Measured)
    [   0  24  59  81 117 119 141 148 180 210 238 247 289 ... 5955 5956 ]
    passed
==== naive scan, power-of-two ====
   elapsed time: 0.074752ms    (CUDA Measured)
    passed
==== naive scan, non-power-of-two ====
   elapsed time: 0.057344ms    (CUDA Measured)
    passed
==== work-efficient scan, power-of-two ====
   elapsed time: 0.304128ms    (CUDA Measured)
    passed
==== work-efficient scan, non-power-of-two ====
   elapsed time: 0.262144ms    (CUDA Measured)
    passed
==== thrust scan, power-of-two ====
   elapsed time: 13.9612ms    (CUDA Measured)
    passed
==== thrust scan, non-power-of-two ====
   elapsed time: 1.49094ms    (CUDA Measured)
    passed

*****************************
** STREAM COMPACTION TESTS **
*****************************
    [   0   0   0   1   2   1   2   1   1   2   1   2   3 ...   2   0 ]
==== cpu compact without scan, power-of-two ====
   elapsed time: 0.0049ms    (std::chrono Measured)
    [   1   2   1   2   1   1   2   1   2   3   2   1   2 ...   3   2 ]
    passed
==== cpu compact without scan, non-power-of-two ====
   elapsed time: 0.0038ms    (std::chrono Measured)
    [   1   2   1   2   1   1   2   1   2   3   2   1   2 ...   3   3 ]
    passed
==== cpu compact with scan ====
   elapsed time: 0.0182ms    (std::chrono Measured)
    [   1   2   1   2   1   1   2   1   2   3   2   1   2 ...   3   2 ]
    passed
==== work-efficient compact, power-of-two ====
   elapsed time: 0.251904ms    (CUDA Measured)
    passed
==== work-efficient compact, non-power-of-two ====
   elapsed time: 0.294912ms    (CUDA Measured)
    passed
