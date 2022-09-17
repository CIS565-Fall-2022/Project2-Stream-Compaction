#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"
#include "iostream"

#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        /*! Block size used for CUDA kernel launch. */
        #define blockSize 128

        // fix this and use new function1
        __global__ void kernReductionHelper(int n, int offset, int* tdata) {
            int idx = threadIdx.x + (blockIdx.x * blockDim.x);
            if (idx < n && (idx + 1) % offset == 0) {
                int neighborLoc = offset / 2;
                int a = tdata[idx - neighborLoc];
                int b = tdata[idx];
                tdata[idx] = a + b;
            }
        }

        // use function from class
        __global__ void kernPartialSumHelper(int n, int offset, int* tdata) {
            int idx = threadIdx.x + (blockIdx.x * blockDim.x);
            if (idx < n && (idx + 1) % offset == 0) {
                int neighborLoc = offset / 2;
                int a = tdata[idx - neighborLoc];
                int b = tdata[idx];

                tdata[idx - neighborLoc] = b;
                tdata[idx] = a + b;
            }
        }

        // shift all nums one to the left
        __global__ void kernShiftLeft(int n, int* odata, int* idata) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index < n) {
                if (index > 0) {
                    odata[index - 1] = idata[index];
                }
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startGpuTimer();

            int fullBlocksPerArray = (n + blockSize - 1) / blockSize;

            // shift tidata to the right to prepend 0's.
            int nextPowTwo = ilog2ceil(n);
            int numZeroes = pow(2, nextPowTwo) - n;

            // 1. up sweep same as reduction
            // empty buffer as idata
            // malloc enough space for n and 0's
            int* dev_tidata;
            cudaMalloc((void**)&dev_tidata, (n + numZeroes) * sizeof(int));
            checkCUDAErrorWithLine("cudaMalloc dev_tifailed failed!");

            // set all elems to 0
            cudaMemset(dev_tidata, 0, (n + numZeroes) * sizeof(int));
            checkCUDAErrorWithLine("cudaMemset all elems to 0 failed!");

            // copy contents of idata into tidata so we can just pass in tidata and modify that on every pass.
            // antyhing after n + numZeroes is all 0's
            cudaMemcpy(dev_tidata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAErrorWithLine("cudaMemcpy idata to dev_tidata failed!");

            int depth = 0;

            for (depth = 1; depth <= ilog2ceil(n + numZeroes); depth++) {
                int offset = pow(2, depth);
                kernReductionHelper << <fullBlocksPerArray, blockSize>>>(n + numZeroes, offset, dev_tidata);
                // wait for cuda timer. wait for all threads to finish
                cudaDeviceSynchronize();
            }

            // set last int of array to 0
            cudaMemset(&dev_tidata[n + numZeroes - 1], 0, sizeof(int));
            checkCUDAErrorWithLine("cudaMemset last int failed!");

            // 2. down sweep
            // takes dev_tidata as input
            for (depth; depth >= 1; depth--) {
                int offset = pow(2, depth);
                kernPartialSumHelper << <fullBlocksPerArray, blockSize >> > (n + numZeroes, offset, dev_tidata);
                cudaDeviceSynchronize();
            }

            // create output array for shifting
            int* dev_tidataFinal;
            cudaMalloc((void**)&dev_tidataFinal, (n + numZeroes) * sizeof(int));
            checkCUDAErrorWithLine("cudaMalloc dev_tidataFinal failed!");

            // copy final result to odata
            cudaMemcpy(odata, dev_tidata, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAErrorWithLine("cudaMemcpy dev_tidata to odata failed!");

            // free all buffers
            cudaFree(dev_tidata);
            cudaFree(dev_tidataFinal);

            timer().endGpuTimer();
        }

        __global__ void kernMapToBoolean(int n, int* odata, int* idata) {
            int idx = threadIdx.x + (blockIdx.x * blockDim.x);
            if (idx < n) {
                if (idata[idx] != 0) {
                    // add 1 to out array
                    odata[idx] = 1;
                }
                else {
                    // add 0 to outarray
                    odata[idx] = 0;
                }
            }
        }

        __global__ void kernScatter(int n, int* odata, int* idata, int* tdata, int* sdata) {
            int idx = threadIdx.x + (blockIdx.x * blockDim.x);
            if (idx < n) {
                if (tdata[idx] == 1) {
                    int destinationIdx = sdata[idx];
                    odata[destinationIdx] = idata[idx];
                }
                // otherwise do not write
            }
        }

        // reimplement scan for compact
        void compactScan(int n, int* dev_odata, int* dev_idata) {
            int fullBlocksPerArray = (n + blockSize - 1) / blockSize;

            // shift tidata to the right to prepend 0's.
            int nextPowTwo = ilog2ceil(n);
            int numZeroes = pow(2, nextPowTwo) - n;

            // 1. up sweep same as reduction
            // empty buffer as idata
            // malloc enough space for n and 0's
            int* dev_tidata;
            cudaMalloc((void**)&dev_tidata, (n + numZeroes) * sizeof(int));
            checkCUDAErrorWithLine("cudaMalloc dev_tifailed failed!");

            // set all elems to 0
            cudaMemset(dev_tidata, 0, (n + numZeroes) * sizeof(int));
            checkCUDAErrorWithLine("cudaMemset all elems to 0 failed!");

            // copy contents of idata into tidata so we can just pass in tidata and modify that on every pass.
            // antyhing after n + numZeroes is all 0's
            cudaMemcpy(dev_tidata, dev_idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAErrorWithLine("cudaMemcpy idata to dev_tidata failed!");

            int depth = 0;

            for (depth = 1; depth <= ilog2ceil(n + numZeroes); depth++) {
                int offset = 1 << depth; // pow(2, depth);
                kernReductionHelper << <fullBlocksPerArray, blockSize >> > (n + numZeroes, offset, dev_tidata);
                // wait for cuda timer. wait for all threads to finish
                cudaDeviceSynchronize();
            }

            // set last int of array to 0
            cudaMemset(&dev_tidata[n + numZeroes - 1], 0, sizeof(int));
            checkCUDAErrorWithLine("cudaMemset last int failed!");

            // 2. down sweep
            // takes dev_tidata as input
            for (depth; depth >= 1; depth--) {
                int offset = pow(2, depth);
                kernPartialSumHelper << <fullBlocksPerArray, blockSize >> > (n + numZeroes, offset, dev_tidata);
                cudaDeviceSynchronize();
            }

            // copy final result to odata
            cudaMemcpy(dev_odata, dev_tidata, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAErrorWithLine("cudaMemcpy dev_tidata to odata failed!");

            // free all buffers
            cudaFree(dev_tidata);
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

            int fullBlocksPerArray = (n + blockSize - 1) / blockSize;

            // 1. compute temp array: 1 for everything that fits rule. 0 otherwise.
            int* dev_iArray;
            int* dev_tempArray;

            cudaMalloc((void**)&dev_tempArray, n * sizeof(int));
            checkCUDAErrorWithLine("cudaMalloc dev_tempArray failed!");

            cudaMalloc((void**)&dev_iArray, n * sizeof(int));
            checkCUDAErrorWithLine("cudaMalloc dev_iArray failed!");

            cudaMemcpy(dev_iArray, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAErrorWithLine("cudaMemcpy dev_iArray failed!");

            kernMapToBoolean << <fullBlocksPerArray, blockSize >> > (n, dev_tempArray, dev_iArray);

            // 2. exclusive scan on tempArray.
            int* dev_scanArray;

            cudaMalloc((void**)&dev_scanArray, n * sizeof(int));
            checkCUDAErrorWithLine("cudaMalloc dev_scanArray failed!");

            compactScan(n, dev_scanArray, dev_tempArray);

            // 3. scatter
            // last element of numScatters is the length of scatterArray.
            int numScatters = 0;
            int validSlot = 0;
            cudaMemcpy(&numScatters, dev_scanArray + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&validSlot, dev_tempArray + n - 1, sizeof(int), cudaMemcpyDeviceToHost);

            numScatters += validSlot;
            checkCUDAErrorWithLine("cudaMemcpy numScatters failed!");

            int* dev_scatterFinal;
            cudaMalloc((void**)&dev_scatterFinal, numScatters * sizeof(int));
            checkCUDAErrorWithLine("cudaMalloc dev_scatterFinal failed!");

            kernScatter << <fullBlocksPerArray, blockSize >> > (n, dev_scatterFinal, dev_iArray, dev_tempArray, dev_scanArray);

            // memcpy back from odata1 to odata
            cudaMemcpy(odata, dev_scatterFinal, numScatters * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAErrorWithLine("cudaMemcpy dev_scatterFinal to odata failed!");

            timer().endGpuTimer();

            cudaFree(dev_iArray);
            cudaFree(dev_tempArray);
            cudaFree(dev_scanArray);
            cudaFree(dev_scatterFinal);

            return numScatters;
        }
    }
}
