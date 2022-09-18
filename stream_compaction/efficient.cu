#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"
#include <device_launch_parameters.h>
#include <iostream>


namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        
        constexpr unsigned blockSize = 128; // TODO test different blockSizes

        __global__ void kernUpsweep(int d, int n, int* data) {
            unsigned index = (blockIdx.x * blockDim.x) + threadIdx.x;
            unsigned rightPOT = 1 << (d + 1);
            index *= rightPOT; // "by 2^(d+1)"
            unsigned rightIdx = index + rightPOT - 1;
            if (rightIdx > n) { return; }
            data[rightIdx] += data[index + (1 << d) - 1];
        }

        __global__ void kernDownsweep(int d, int n, int* data) {
            unsigned index = (blockIdx.x * blockDim.x) + threadIdx.x;
            unsigned rightPOT = 1 << (d + 1);
            index *= rightPOT;
            unsigned leftIdx = index + (1 << d) - 1;
            unsigned rightIdx = index + rightPOT - 1;
            if (rightIdx > n) { return; }

            int tmp = data[leftIdx];
            data[leftIdx] = data[rightIdx];
            data[rightIdx] += tmp;
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
            int smallestPOTGreater = 1 << ilog2ceil(n); // smallest POT larger than n

            int* dev_data;
            cudaMalloc((void**)&dev_data, smallestPOTGreater * sizeof(int));
            cudaMemcpy(dev_data, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            cudaMemset(dev_data + n, 0, (smallestPOTGreater - n) * sizeof(int)); // necessary? 

            int neededThreads = smallestPOTGreater;

            dim3 fullBlocksPerGrid;
            for (int d = 0; d < ilog2ceil(smallestPOTGreater); ++d, neededThreads /= 2) {
                fullBlocksPerGrid = (neededThreads + blockSize - 1) / blockSize;
                kernUpsweep<<<fullBlocksPerGrid, blockSize>>>(d, n, dev_data);
                cudaDeviceSynchronize();
            }

            cudaMemset(&dev_data[smallestPOTGreater - 1], 0, sizeof(int));

            for (int d = ilog2ceil(smallestPOTGreater) - 1; d >= 0; --d, neededThreads *= 2) {
                fullBlocksPerGrid = (neededThreads + blockSize - 1) / blockSize;
                kernDownsweep<<<fullBlocksPerGrid, blockSize>>>(d, smallestPOTGreater, dev_data);
                cudaDeviceSynchronize();
            }

            cudaMemcpy(odata, dev_data, n * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(dev_data);
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
