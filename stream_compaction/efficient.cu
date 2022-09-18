#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"
#include <iostream>

#define blockSize 128 

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        __global__ void kernUpsweep(int n, int depth, int* odata) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            int path = 1 << depth;
            if (index >= n || index % (2*path)) {
                return;
            }
            odata[index + 2*path - 1] += odata[index + path - 1];
            return;
        }

        __global__ void kernDownsweep(int n, int depth, int* odata) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            int path = 1<< depth;
            if (index >= n || index % (2 * path)) {
                return;
            }
            int saveVal = odata[index + path - 1];
            odata[index + path - 1] = odata[index + 2 * path - 1];
            odata[index + 2 * path - 1] += saveVal;
            return;
        }

        inline void scanImpl(int maxN, int* odataMax) {
            dim3 blockDim((maxN + blockSize - 1) / blockSize);
            for (int i = 0; i < ilog2ceil(maxN); ++i) {
                kernUpsweep << <blockDim, blockSize >> > (maxN, i, odataMax);
                checkCUDAError("kernUpSweep failed");
            }
            int addr = 0;
            //set the first value to 0 by using MemCpy
            cudaMemcpy(odataMax + maxN - 1, &addr, sizeof(int), cudaMemcpyHostToDevice);
            //Downsweep
            for (int i = ilog2ceil(maxN) - 1; i >= 0; --i) {
                kernDownsweep << <blockDim, blockSize >> > (maxN, i, odataMax);
                checkCUDAError("kernDownSweep failed");
            }
        }

        inline void fillArray(int** odataMax, const int* idata, int maxN, int n) {
            cudaMalloc(odataMax, maxN * sizeof(int));
            cudaMemcpy(*odataMax, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            if (n != maxN) {
                cudaMemset(*odataMax + n, 0, (maxN - n) * sizeof(int));
            }
        }
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            //fill array and initialization
            int maxN = pow(2, ilog2ceil(n));
            int* odataMax = nullptr;
            fillArray(&odataMax, idata, maxN, n);
            /*cudaMalloc((void**)&odataMax, maxN * sizeof(int));
            cudaMemcpy(odataMax, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            if (n != maxN) {
                cudaMemset(odataMax + n, 0, (maxN - n) * sizeof(int));
            }*/

            timer().startGpuTimer();
            //encapsulation for radixsort later
            scanImpl(maxN, odataMax);
            timer().endGpuTimer();

            cudaMemcpy(odata, odataMax, n*sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(odataMax);
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
            //initialization
            int maxN = 1 << ilog2ceil(n);
            dim3 blockDim((n + blockSize - 1) / blockSize);
            int* odataMax; 
            int* oBoll;
            int *oScan = nullptr;
            int* dev_out;

            cudaMalloc((void**)&odataMax, n * sizeof(int));
            cudaMemcpy(odataMax, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            cudaMalloc((void**)&dev_out, n * sizeof(int));
            cudaMalloc((void**)&oBoll, n * sizeof(int));
            cudaMalloc((void**)&oScan, maxN * sizeof(int));

            timer().startGpuTimer();
            // TODO
            //first create temp array
            StreamCompaction::Common::kernMapToBoolean << <blockDim, blockSize >> > (n, oBoll, odataMax);
            cudaMemcpy(oScan, oBoll, n * sizeof(int), cudaMemcpyDeviceToDevice);
            cudaMemset(oScan + n, 0, (maxN - n) * sizeof(int));
            scanImpl(maxN, oScan);
            //scatter
            StreamCompaction::Common::kernScatter << <blockDim, blockSize >> > (n, dev_out, odataMax, oBoll, oScan);
            timer().endGpuTimer();

            //now we get the last index of oScan to return;
            int lastIndex1;
            cudaMemcpy(&lastIndex1, oScan + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            int lastIndex2;
            cudaMemcpy(&lastIndex2, oBoll + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            int lastIndex = lastIndex1 + lastIndex2;
            cudaMemcpy(odata, dev_out, lastIndex*sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(oBoll);
            cudaFree(oScan);
            cudaFree(odataMax);
            cudaFree(dev_out);
            return lastIndex;//how do we get the final index?
        }
    }
}
