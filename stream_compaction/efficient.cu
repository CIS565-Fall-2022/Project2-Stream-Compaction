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
        __global__ void reductionHelper(int n, int offset, int* tdata) {
            int idx = threadIdx.x + (blockIdx.x * blockDim.x);
            if (idx < n && (idx + 1) % offset == 0) {
                int neighborLoc = offset / 2;
                int a = tdata[idx - neighborLoc];
                int b = tdata[idx];
                tdata[idx] = a + b;

                //printf("red: %i, \n", a + b);
            }
        }

        // use function from class
        __global__ void partialSumHelper(int n, int offset, int* tdata) {
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
        __global__ void shiftLeft(int n, int* odata, int* idata) {
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
                reductionHelper << <fullBlocksPerArray, blockSize>>>(n + numZeroes, offset, dev_tidata);
                // wait for cuda timer. wait for all threads to finish
                cudaDeviceSynchronize();
            }

            // store the last int of the array temporarily
            int lastInt = 0;
            cudaMemcpy(&lastInt, dev_tidata + n + numZeroes - 1, sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAErrorWithLine("cudaMemcpy lastInt Device to Host failed!");

            // set last int of array to 0
            cudaMemset(&dev_tidata[n + numZeroes - 1], 0, sizeof(int));
            checkCUDAErrorWithLine("cudaMemset last int failed!");

            // 2. down sweep
            // takes dev_tidata as input
            for (depth; depth >= 1; depth--) {
                int offset = pow(2, depth);
                partialSumHelper << <fullBlocksPerArray, blockSize >> > (n + numZeroes, offset, dev_tidata);
                cudaDeviceSynchronize();
            }

            // create output array for shifting
            int* dev_tidataFinal;
            cudaMalloc((void**)&dev_tidataFinal, (n + numZeroes) * sizeof(int));
            checkCUDAErrorWithLine("cudaMalloc dev_tidataFinal failed!");

            // shift entire list to the left by 1 to remove extraneous 0 at the beginning
            shiftLeft << <fullBlocksPerArray, blockSize >> > (n + numZeroes, dev_tidataFinal, dev_tidata);

            // set last number in the array
            cudaMemcpy(dev_tidata + n + numZeroes - 1, &lastInt, sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAErrorWithLine("cudaMemcpy lastInt Host to Device failed!");

            // copy final result to odata
            cudaMemcpy(odata, dev_tidata, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAErrorWithLine("cudaMemcpy dev_tidata to odata failed!");

            // free all buffers
            cudaFree(dev_tidata);
            cudaFree(dev_tidataFinal);

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

            // 1. compute temp array
            
            // 2. exclusive scan
            // 3. scatter
            
            timer().endGpuTimer();
            return -1;
        }
    }
}
