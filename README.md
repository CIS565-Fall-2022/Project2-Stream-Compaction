CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Shutong Wu
  * [LinkedIn](https://www.linkedin.com/in/shutong-wu-214043172/)
  * [Email](shutong@seas.uepnn.edu)
* Tested on: Windows 10, i7-10700K CPU @ 3.80GHz, RTX3080, SM8.6, Personal Computer 

## GPU Scan and Stream Compaction
Implementation including:
- CPU Version of Scan
- CPU Version of Compact with Scan and without Scan
- GPU Naive Version
- GPU Work-Efficient Scan
- GPU Work-Effient Compact
- CUDA Thrust Library Scan and Compaction
- Radix Sort(Extra Credit)
TODO: Shared Memory Optimization

## Performance Analysis
- We use runtime to measure the performance of each version of scan/compaction, the less time it took to run, the better the performance.
- We will test with arrays that have sizes from 2^8 to 2^19, and compare the performances of different types of scans and compacts
- All number varies from 0 to 99, and the run time is mesasured in ms;
- NPOT in hte following charts means Non Power of Two

### Performance Analysis of Scan

### Performance Analysis of Compaction

### Performance Analysis of Radix Sort


### Why is My GPU Approach So Slow?


Include analysis, etc. (Remember, this is public, so don't put
anything here that you don't want to share with the world.)

