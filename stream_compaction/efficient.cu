#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        __global__ void kernUpSweep(int n, int d, int* idata) {
            // Parallel Reduction
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n) {
                return;
            }

            int k = index * (1 << (d + 1));
            idata[k + (1 << (d + 1)) - 1] += idata[k + (1 << d) - 1];
        }

        __global__ void kernDownSweep(int n, int d, int* idata) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n) {
                return;
            }

            int k = index * (1 << (d + 1));
            int t = idata[k + (1 << d) - 1];
            idata[k + (1 << d) - 1] = idata[k + (1 << (d + 1)) - 1];
            idata[k + (1 << (d + 1)) - 1] += t;
        }

        __global__ void kernZeroRoot(int n, int* idata) {
            // Root is last element
            idata[n - 1] = 0;
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            // Account for non-powers of 2 by padding by 0
            int paddedN = (1 << ilog2ceil(n));
            int* dev_idata;
            cudaMalloc((void**)&dev_idata, paddedN * sizeof(int));
            checkCUDAError("cudaMalloc dev_idata failed!");
            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemset(dev_idata + n, 0, (paddedN - n) * sizeof(int));
            cudaDeviceSynchronize();

            timer().startGpuTimer();
            // Upsweep
            for (int i = 0; i < ilog2ceil(n); ++i) {
                int numThreads = paddedN / (1 << (i + 1));
                dim3 upSweepGridSize((numThreads + blockSize - 1) / blockSize);
                kernUpSweep << <upSweepGridSize, blockSize >> >
                    (numThreads, i, dev_idata);
                checkCUDAError("kernUpSweep failed!");
                cudaDeviceSynchronize();
            }

            // Downsweep
            kernZeroRoot << <1, 1 >> > (paddedN, dev_idata);
            for (int i = ilog2ceil(n) - 1; i >= 0; --i) {
                int numThreads = paddedN / (1 << (i + 1));
                dim3 downSweepGridSize((numThreads + blockSize - 1) / blockSize);
                kernDownSweep << <downSweepGridSize, blockSize >> >
                    (numThreads, i, dev_idata);
                checkCUDAError("kernDownSweep failed!");
                cudaDeviceSynchronize();
            }
            timer().endGpuTimer();
            cudaMemcpy(odata, dev_idata, n * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(dev_idata);
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
            // Account for non-powers of 2 by padding by 0
            int paddedN = (1 << ilog2ceil(n));
            int* dev_idata;
            int* dev_odata;
            int* dev_bool;
            int* dev_indices;

            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_idata failed!");
            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            
            // Pad bool array instead of idata to save operations in kernMapToBoolean
            cudaMalloc((void**)&dev_bool, paddedN * sizeof(int));
            checkCUDAError("cudaMalloc dev_bool failed!");
            cudaMemset(dev_bool + n, 0, (paddedN - n) * sizeof(int));

            cudaMalloc((void**)&dev_indices, paddedN * sizeof(int));
            checkCUDAError("cudaMalloc dev_indices failed!");
            cudaMalloc((void**)&dev_odata, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_indices failed!");
            cudaDeviceSynchronize();

            timer().startGpuTimer();
            // Binarize
            dim3 nGridSize((n + blockSize - 1) / blockSize);
            StreamCompaction::Common::kernMapToBoolean << < nGridSize, blockSize >> >
                (n, dev_bool, dev_idata);
            checkCUDAError("kernMapToBoolean failed!");
            cudaDeviceSynchronize();
            // We need bool array for scatter so copy bool result to indices to be modified in place
            cudaMemcpy(dev_indices, dev_bool, paddedN * sizeof(int), cudaMemcpyDeviceToDevice);
            checkCUDAError("cudaMemcpy failed!");
            cudaDeviceSynchronize();
 
            // Copied Scan code from above
            // Upsweep
            for (int i = 0; i < ilog2ceil(n); ++i) {
                int numThreads = paddedN / (1 << (i + 1));
                dim3 upSweepGridSize((numThreads + blockSize - 1) / blockSize);
                kernUpSweep << <upSweepGridSize, blockSize >> >
                    (numThreads, i, dev_indices);
                checkCUDAError("kernUpSweep failed!");
                cudaDeviceSynchronize();
            }

            // Downsweep
            kernZeroRoot << <1, 1 >> > (paddedN, dev_indices);
            for (int i = ilog2ceil(n) - 1; i >= 0; --i) {
                int numThreads = paddedN / (1 << (i + 1));
                dim3 downSweepGridSize((numThreads + blockSize - 1) / blockSize);
                kernDownSweep << <downSweepGridSize, blockSize >> >
                    (numThreads, i, dev_indices);
                checkCUDAError("kernDownSweep failed!");
                cudaDeviceSynchronize();
            }

            // Scatter
            StreamCompaction::Common::kernScatter << <nGridSize, blockSize >> >
                (n, dev_odata, dev_idata, dev_bool, dev_indices);
            checkCUDAError("kernScatter failed!");
            cudaDeviceSynchronize();
            timer().endGpuTimer();
            
            cudaMemcpy(odata, dev_indices, paddedN * sizeof(int), cudaMemcpyDeviceToHost);
            int finalNum = odata[paddedN - 1];
            cudaMemcpy(odata, dev_odata, finalNum * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(dev_idata);
            cudaFree(dev_bool);
            cudaFree(dev_indices);
            cudaFree(dev_odata);
            return finalNum;
        }
    }
}
