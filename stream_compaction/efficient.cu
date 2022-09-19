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
        #define blockSizeEffScan 256
        #define blockSizeCompact 256

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
            int fullBlocksPerArray = (n + blockSizeEffScan - 1) / blockSizeEffScan;

            // for performance analysis
            printf("blockSize: %i \n", blockSizeEffScan);

            // shift tidata to the right to prepend 0's.
            int nextPowTwo = ilog2ceil(n);
            int numZeroes = pow(2, nextPowTwo) - n;

            // 1. up sweep same as reduction
            // empty buffer as idata
            // malloc enough space for n and 0's
            int* dev_tidata;
            cudaMalloc((void**)&dev_tidata, (n + numZeroes) * sizeof(int));
            checkCUDAErrorWithLine("cudaMalloc dev_tifailed failed!");

            // create output array for shifting
            int* dev_tidataFinal;
            cudaMalloc((void**)&dev_tidataFinal, (n + numZeroes) * sizeof(int));
            checkCUDAErrorWithLine("cudaMalloc dev_tidataFinal failed!");

            // set all elems to 0
            cudaMemset(dev_tidata, 0, (n + numZeroes) * sizeof(int));
            checkCUDAErrorWithLine("cudaMemset all elems to 0 failed!");

            // copy contents of idata into tidata so we can just pass in tidata and modify that on every pass.
            // antyhing after n + numZeroes is all 0's
            cudaMemcpy(dev_tidata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAErrorWithLine("cudaMemcpy idata to dev_tidata failed!");

            // start gpu timer
            timer().startGpuTimer();

            int depth = 0;

            for (depth = 1; depth <= ilog2ceil(n + numZeroes); depth++) {
                int offset = pow(2, depth);
                kernReductionHelper << <fullBlocksPerArray, blockSizeEffScan>>>(n + numZeroes, offset, dev_tidata);
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
                kernPartialSumHelper << <fullBlocksPerArray, blockSizeEffScan >> > (n + numZeroes, offset, dev_tidata);
                cudaDeviceSynchronize();
            }

            // stop gpu timer
            timer().endGpuTimer();

            // copy final result to odata
            cudaMemcpy(odata, dev_tidata, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAErrorWithLine("cudaMemcpy dev_tidata to odata failed!");

            // free all buffers
            cudaFree(dev_tidata);
            cudaFree(dev_tidataFinal);
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
            int fullBlocksPerArray = (n + blockSizeCompact - 1) / blockSizeCompact;

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
                kernReductionHelper << <fullBlocksPerArray, blockSizeCompact >> > (n + numZeroes, offset, dev_tidata);
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
                kernPartialSumHelper << <fullBlocksPerArray, blockSizeCompact >> > (n + numZeroes, offset, dev_tidata);
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
            int fullBlocksPerArray = (n + blockSizeCompact - 1) / blockSizeCompact;

            printf("blockSize: %i \n", blockSizeCompact);
            // 1. compute temp array: 1 for everything that fits rule. 0 otherwise.
            int* dev_iArray;
            int* dev_tempArray;

            cudaMalloc((void**)&dev_tempArray, n * sizeof(int));
            checkCUDAErrorWithLine("cudaMalloc dev_tempArray failed!");

            cudaMalloc((void**)&dev_iArray, n * sizeof(int));
            checkCUDAErrorWithLine("cudaMalloc dev_iArray failed!");

            cudaMemcpy(dev_iArray, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAErrorWithLine("cudaMemcpy dev_iArray failed!");

            int* dev_scanArray;
            cudaMalloc((void**)&dev_scanArray, n * sizeof(int));
            checkCUDAErrorWithLine("cudaMalloc dev_scanArray failed!");

            // start gpu timer
            timer().startGpuTimer();

            kernMapToBoolean << <fullBlocksPerArray, blockSizeCompact >> > (n, dev_tempArray, dev_iArray);

            // 2. exclusive scan on tempArray.
            compactScan(n, dev_scanArray, dev_tempArray);

            // 3. scatter
            // last element of numScatters is the length of scatterArray.
            int numScatters = 0;
            int validSlot = 0;
            cudaMemcpy(&numScatters, dev_scanArray + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&validSlot, dev_tempArray + n - 1, sizeof(int), cudaMemcpyDeviceToHost);

            numScatters += validSlot;
            checkCUDAErrorWithLine("cudaMemcpy numScatters failed!");

            // unavoidable malloc
            int* dev_scatterFinal;
            cudaMalloc((void**)&dev_scatterFinal, numScatters * sizeof(int));
            checkCUDAErrorWithLine("cudaMalloc dev_scatterFinal failed!");

            kernScatter << <fullBlocksPerArray, blockSizeCompact >> > (n, dev_scatterFinal, dev_iArray, dev_tempArray, dev_scanArray);

            // end gpu timer
            timer().endGpuTimer();

            // memcpy back from odata1 to odata
            cudaMemcpy(odata, dev_scatterFinal, numScatters * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAErrorWithLine("cudaMemcpy dev_scatterFinal to odata failed!");


            cudaFree(dev_iArray);
            cudaFree(dev_tempArray);
            cudaFree(dev_scanArray);
            cudaFree(dev_scatterFinal);

            return numScatters;
        }
    }
}

namespace RadixSort {
    //// todo import performance timer

    /*! Block size used for CUDA kernel launch. */
    #define blockSize 128

    __global__ void kernMapToBoolean(int n, int* odata, int* idata, int bitpos) {
        int idx = threadIdx.x + (blockIdx.x * blockDim.x);
        if (idx < n) {
            int num = idata[idx];
            int bit = (num & (1 << bitpos)) >> bitpos;
            // printf("num: %i, bitpos: %i, bit: %i \n", num, bitpos, bit);

            if (bit != 0) {
                // add 1 to out array
                odata[idx] = 1;
                // printf("idx: %i, val: %i \n", idx, 1);
            }
            else {
                // add 0 to outarray
                odata[idx] = 0;
                // printf("idx: %i, val: %i \n", idx, 0);
            }
        }
    }

    __global__ void kernComputeScatter(int n, int bitpos, int* ddata, int* bdata, int* tdata, int* fdata, int* idata) {
        int idx = threadIdx.x + (blockIdx.x * blockDim.x);
        if (idx < n) {
            if (bdata[idx] == 1) {
                int numToScatter = idata[idx];
                int dest = tdata[idx];
                ddata[dest] = idata[idx];
                // printf("bitpos: %i, scatter num: %i, destination: %i \n", bitpos, numToScatter, dest);
            }
        }
    }

    __global__ void kernComputeT(int n, int* tdata, int* fdata, int totalFalses) {
        int idx = threadIdx.x + (blockIdx.x * blockDim.x);
        if (idx < n) {
            tdata[idx] = idx - fdata[idx] + totalFalses;
        }

    }

    __global__ void kernComputeE(int n, int* odata, int* idata) {
        int idx = threadIdx.x + (blockIdx.x * blockDim.x);
        if (idx < n) {
            odata[idx] = !idata[idx];
        }
    }

    void split(int n, int* odata, int* idata, int bitpos) {
        // the following operations should only be done on the bit specified by bitpos.
        // todo implement radix sort
        int fullBlocksPerArray = (n + blockSize - 1) / blockSize;

        // 1. compute e array
        int* dev_iarray;
        cudaMalloc((void**)&dev_iarray, n * sizeof(int));
        checkCUDAErrorWithLine("cudaMalloc dev_iarray failed!");
        cudaMemcpy(dev_iarray, idata, n * sizeof(int), cudaMemcpyHostToDevice);
        checkCUDAErrorWithLine("cudaMemcpy dev_iarray failed!");

        int* dev_barray;
        cudaMalloc((void**)&dev_barray, n * sizeof(int));
        checkCUDAErrorWithLine("cudaMalloc dev_barray failed!");

        // call kernMapToBoolean to calculate b
        kernMapToBoolean<<<fullBlocksPerArray, blockSize>>>(n, dev_barray, dev_iarray, bitpos);

        //int* tempBuf = new int[n];
        //cudaMemcpy(tempBuf, dev_barray, n * sizeof(int), cudaMemcpyDeviceToHost);
        //checkCUDAErrorWithLine("cudaMempy dev_tarray failed!");

        //for (int i = 0; i < n; i++) {
        //    std::cout << "t: " << bitpos << " " << tempBuf[i] << std::endl;
        //}

        int* dev_earray;
        cudaMalloc((void**)&dev_earray, n * sizeof(int));
        checkCUDAErrorWithLine("cudaMalloc dev_earray failed!");

        kernComputeE << <fullBlocksPerArray, blockSize >> > (n, dev_earray, dev_barray);

        // 2. exclusive scan on e and store in f
        int* dev_farray;
        cudaMalloc((void**)&dev_farray, n * sizeof(int));
        checkCUDAErrorWithLine("cudaMalloc dev_farray failed!");

        // use compact scan because it doesn't presume the input is a host pointer.
        StreamCompaction::Efficient::compactScan(n, dev_farray, dev_earray);

        // 3. compute total # falses
        int fArrayLast = 0;
        int eArrayLast = 0;
        cudaMemcpy(&fArrayLast, dev_farray + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
        checkCUDAErrorWithLine("cudaMemcpy fArrayLast failed!");
        cudaMemcpy(&eArrayLast, dev_earray + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
        checkCUDAErrorWithLine("cudaMemcpy eArrayLast failed!");

        int totalFalses = fArrayLast + eArrayLast;
        
        // 4. compute t array which contains the address for writing data.
        int* dev_tarray;
        cudaMalloc((void**)&dev_tarray, n * sizeof(int));
        checkCUDAErrorWithLine("cudaMalloc dev_tarray failed!");

        kernComputeT << <fullBlocksPerArray, blockSize >> > (n, dev_tarray, dev_farray, totalFalses);

        // 5. scatter based on address d.
        int* dev_scatterFinal; // final output. send to odata
        cudaMalloc((void**)&dev_scatterFinal, n * sizeof(int));
        checkCUDAErrorWithLine("cudaMalloc dev_scatterFinal failed!");

        kernComputeScatter << <fullBlocksPerArray, blockSize >> > (n, bitpos, dev_scatterFinal, dev_barray, dev_tarray, dev_farray, dev_iarray);

        // memcpy back to odata
        cudaMemcpy(odata, dev_scatterFinal, n * sizeof(int), cudaMemcpyDeviceToHost);

        // delete[] tempBuf;
    }

    void radixSort(int n, int* odata, int* idata) {
        int numbits = 3;
        for (int i = 0; i < numbits; i++) {
            split(n, odata, idata, i);
        }
    }
}

