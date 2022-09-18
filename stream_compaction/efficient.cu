#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

#define blockSize 128

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        // unlike naive impl, this one doesn't shift the array
        __global__ void kernPadArray(int n, int paddedLen, int* odata, const int* idata) {
          int index = threadIdx.x + (blockIdx.x * blockDim.x);
          if (index < n) {
            odata[index] = idata[index];
          }
          else if (index < paddedLen) {
            odata[index] = 0;
          }
        }

        __global__ void kernUpsweep(int numThreads, int readStride, int* data) {
          int index = threadIdx.x + (blockIdx.x * blockDim.x);
          if (index >= numThreads) {
            return;
          }

          int writeStride = readStride * 2;

          // Index of what element to write to is calculated using write stride
          int writeIndex = (writeStride * index) + writeStride - 1;
          int readIndex = (writeStride * index) + readStride - 1;

          data[writeIndex] += data[readIndex];
        }

        __global__ void kernDownsweep(int numThreads, int writeStride, int* data) {
          int index = threadIdx.x + (blockIdx.x * blockDim.x);
          if (index >= numThreads) {
            return;
          }

          int readStride = writeStride * 2;

          int leftChildIndex = index * readStride + writeStride - 1;
          int rightChildIndex = index * readStride + readStride - 1; // right child is also where parent is stored

          int temp = data[leftChildIndex];
          data[leftChildIndex] = data[rightChildIndex];
          data[rightChildIndex] += temp;
        }

        int* dev_unpadded_idata;
        int* dev_idata;
        int* dev_odata;

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
            // TODO
            // Pad array
            int exponent = ilog2ceil(n);
            int paddedLength = pow(2, exponent);
            dim3 fullBlocksPerGrid((paddedLength + blockSize - 1) / blockSize);

            cudaMalloc((void**)&dev_unpadded_idata, n * sizeof(int));
            cudaMalloc((void**)&dev_idata, paddedLength * sizeof(int));

            cudaMemcpy(dev_unpadded_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAErrorFn("Cuda memcpy idata no work");

            kernPadArray << <fullBlocksPerGrid, blockSize >> > (n, paddedLength, dev_idata, dev_unpadded_idata);

            // Build tree (upsweep)
            // readStride = 2^depth, where depth goes from 0... log2n - 1... the stride between elements we read and sum
            // writeStride = 2^(depth + 1)... the stride between indices of elements we store the sums in
            for (int readStride = 1; readStride < paddedLength; readStride *= 2) {
              int writeStride = readStride * 2;

              int numThreads = paddedLength / writeStride; // one thread per element we write to
              dim3 numBlocks((numThreads + blockSize - 1) / blockSize);

              kernUpsweep << <numBlocks, blockSize >> > (numThreads, readStride, dev_idata);
              cudaDeviceSynchronize();
            }

            // Down sweep
            // In down sweep, children now read info from parent. So writeStride = readStride / 2
            // Write stride = n/2, n/4, ... 4, 2, 1, aka. 2^depth
            // Read stride = 2^(depth + 1)
            
            // First set parent to 0
            int zero = 0;
            cudaMemcpy(dev_idata + paddedLength - 1, &zero, sizeof(int), cudaMemcpyHostToDevice);

            for (int writeStride = paddedLength / 2; writeStride >= 1; writeStride = writeStride >> 1) {
              int readStride = writeStride * 2;
              
              // now launch 1 thread per element we read from
              int numThreads = paddedLength / readStride;
              dim3 numBlocks((numThreads + blockSize - 1) / blockSize);

              kernDownsweep << <numBlocks, blockSize >> > (numThreads, writeStride, dev_idata);
            }

            cudaMemcpy(odata, dev_idata, n * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(dev_unpadded_idata);
            cudaFree(dev_idata);

            timer().endGpuTimer();
        }

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
            // TODO
            timer().endGpuTimer();
            return -1;
        }
    }
}
