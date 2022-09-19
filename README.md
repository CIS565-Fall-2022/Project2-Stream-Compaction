CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture,
Project 2 - CUDA Stream Compaction**

* Edward Zhang
  * https://www.linkedin.com/in/edwardjczhang/
  * https://zedward23.github.io/personal_Website/
 
* Tested on: Windows 10 Home, i7-11800H @ 2.3GHz, 16.0GB, NVIDIA GeForce RTX 3060 Laptop GPU

### Background




## Why is My GPU Approach So Slow? (Extra Credit) (+5)

If you implement your efficient scan version following the slides closely, there's a good chance
that you are getting an "efficient" gpu scan that is actually not that efficient -- it is slower than the cpu approach?

Though it is totally acceptable for this assignment,
In addition to explain the reason of this phenomena, you are encouraged to try to upgrade your work-efficient gpu scan.

Thinking about these may lead you to an aha moment:
- What is the occupancy at a deeper level in the upper/down sweep? Are most threads actually working?
- Are you always launching the same number of blocks throughout each level of the upper/down sweep?
- If some threads are being lazy, can we do an early termination on them?
- How can I compact the threads? What should I modify to keep the remaining threads still working correctly?



Include analysis, etc. (Remember, this is public, so don't put
anything here that you don't want to share with the world.)

