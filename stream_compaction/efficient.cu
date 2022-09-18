#include <iostream>
#include <memory>
#include <cuda.h>
#include <device_launch_parameters.h>
#include <device_functions.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer() {
            static PerformanceTimer timer;
            return timer;
        }
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        __global__ void kernUpSweep(int n, int num, int offset, int* inp) {
            int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (idx >= num) return;
            int idxWrite = offset * 2 * (idx + 1) - 1;
            inp[idxWrite] = inp[idxWrite] + inp[idxWrite - offset];
        }

        __global__ void kernDownSweep(int n, int num, int offset, int* inp) {
            int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (idx >= num) return;
            int idxWrite = n - 1 - idx * offset * 2;
            int tmp = inp[idxWrite];
            inp[idxWrite] += inp[idxWrite - offset];
            inp[idxWrite - offset] = tmp;
        }

        void scan(int origN, int *odata, const int *idata, bool enableTimer) {
            int* devInp;
            int log2n = ilog2ceil(origN);

            int n = pow(2, log2n);
            cudaMalloc((void**)&devInp, n * sizeof(int));
            checkCUDAError("cudaMalloc devInp failed!");
            cudaMemcpy(devInp, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy idata failed!");

            if (enableTimer) timer().startGpuTimer();

            // up sweep
            int num = n / 2;
            for (int d = 0; d < log2n; d++) {
                int offset = 1 << d;
                dim3 fullBlocksPerGrid((num + BLOCK_SIZE - 1) / BLOCK_SIZE);
                kernUpSweep<<<fullBlocksPerGrid, BLOCK_SIZE>>>(n, num, offset, devInp);
                num /= 2;
            }
            cudaMemset(devInp+n-1, 0, sizeof(int));

            // down sweep
            int offset = n / 2;
            for (int d = 0; d < log2n; d++) {
                int num = 1 << d;
                dim3 fullBlocksPerGrid((num + BLOCK_SIZE - 1) / BLOCK_SIZE);
                kernDownSweep<<<fullBlocksPerGrid, BLOCK_SIZE>>> (n, num, offset, devInp);
                offset /= 2;
            }

            if (enableTimer) timer().endGpuTimer();

            cudaMemcpy(odata, devInp, origN * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy odata failed!");
            cudaFree(devInp);
            checkCUDAError("cudaFree devInp failed!");
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


        int compact(int origN, int *odata, const int *idata, bool enableTimer) {
            int* devInp;
            int* devBools;
            int* devOut;

            int log2n = ilog2ceil(origN);
            int n = pow(2, log2n);

            cudaMalloc((void**)&devInp, n * sizeof(int));
            checkCUDAError("cudaMalloc devInp failed!");
            cudaMalloc((void**)&devOut, n * sizeof(int));
            checkCUDAError("cudaMalloc devOut failed!");
            cudaMalloc((void**)&devBools, n * sizeof(int));
            checkCUDAError("cudaMalloc devBools failed!");
            cudaMemcpy(devInp, idata, origN * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy idata failed!");

            if (enableTimer) timer().startGpuTimer();
            dim3 fullBlocksPerGrid((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
            Common::kernMapToBoolean<<<fullBlocksPerGrid, BLOCK_SIZE>>> (n, devBools, devInp);

            // up sweep
            int num = n / 2;
            for (int d = 0; d < log2n; d++) {
                int offset = 1 << d;
                dim3 fullBlocksPerGrid((num + BLOCK_SIZE - 1) / BLOCK_SIZE);
                kernUpSweep << <fullBlocksPerGrid, BLOCK_SIZE >> > (n, num, offset, devBools);
                num /= 2;
            }
            cudaMemset(devBools + n - 1, 0, sizeof(int));

            // down sweep
            int offset = n / 2;
            for (int d = 0; d < log2n; d++) {
                int num = 1 << d;
                dim3 fullBlocksPerGrid((num + BLOCK_SIZE - 1) / BLOCK_SIZE);
                kernDownSweep << <fullBlocksPerGrid, BLOCK_SIZE >> > (n, num, offset, devBools);
                offset /= 2;
            }

            Common::kernScatter<<<fullBlocksPerGrid, BLOCK_SIZE>>>(n, devOut, devInp, devBools);

            if (enableTimer) timer().endGpuTimer();

            cudaMemcpy(odata, devOut, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy odata failed!");
            std::unique_ptr<int[]> indices{ new int[n] };
            cudaMemcpy(indices.get(), devBools, origN * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy devBools failed!");
            cudaFree(devInp);
            checkCUDAError("cudaFree devInp failed!");
            cudaFree(devOut);
            checkCUDAError("cudaFree devInp failed!");
            cudaFree(devBools);
            checkCUDAError("cudaFree devBools failed!");
            //for (int i = 0; i < 32; i++) {
            //    std::cout << indices[i] << " ";
            //}
            //std::cout << std::endl;
            return idata[origN - 1] != 0 ? indices[origN - 1] + 1 : indices[origN - 1];
        }

        // radix sort

        __global__ void kernCheckBit(int n, int bit, int* inp, int* booleans, int* invertBooleans) {
            int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (idx >= n) return;
            int boolean = (inp[idx] & (1 << bit)) == 0 ? 0 : 1;
            booleans[idx] = boolean;
            invertBooleans[idx] = boolean == 0 ? 1 : 0;
        }

        __global__ void kernComputeIndices(int n, int totalFalse, int* scannedFalse, int* booleans, int* out) {
            int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (idx >= n) return;
            if (booleans[idx] == 1)
                out[idx] = idx - scannedFalse[idx] + totalFalse;
            else
                out[idx] = scannedFalse[idx];
        }

        __global__ void kernRadixSortScatter(int n, int* indices, int* inp, int* out) {
            int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (idx >= n) return;
            out[indices[idx]] = inp[idx];
        }

        void radixSort(int n, int* out, const int* inp, bool enableTimer) {
            int* devInp;
            int* devTrue;
            int* devFalse;
            int* devIndices;
            int log2n = ilog2ceil(n);
            int nForScan = pow(2, log2n);

            cudaMalloc((void**)&devInp, n * sizeof(int));
            checkCUDAError("cudaMalloc devInp failed!");
            cudaMemcpy(devInp, inp, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy idata failed!");
            cudaMalloc((void**)&devTrue, n * sizeof(int));
            checkCUDAError("cudaMalloc devTrue failed!");
            cudaMalloc((void**)&devIndices, n * sizeof(int));
            checkCUDAError("cudaMalloc devIndices failed!");
            cudaMalloc((void**)&devFalse, nForScan * sizeof(int)); // devFalse will be scanned
            checkCUDAError("cudaMalloc devFalse failed!");

            if (enableTimer) timer().startGpuTimer();

            for (int bit = 0; bit < 32; bit++) {
                dim3 fullBlocksPerGrid((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
                kernCheckBit <<<fullBlocksPerGrid, BLOCK_SIZE>>> (n, bit, devInp, devTrue, devFalse);

                {// scan devFalse
                    int num = nForScan / 2;
                    for (int d = 0; d < log2n; d++) {
                        int offset = 1 << d;
                        dim3 fullBlocksPerGrid((num + BLOCK_SIZE - 1) / BLOCK_SIZE);
                        kernUpSweep <<<fullBlocksPerGrid, BLOCK_SIZE >>> (nForScan, num, offset, devFalse);
                        num /= 2;
                    }
                    cudaMemset(devFalse + nForScan - 1, 0, sizeof(int));

                    // down sweep
                    int offset = nForScan / 2;
                    for (int d = 0; d < log2n; d++) {
                        int num = 1 << d;
                        dim3 fullBlocksPerGrid((num + BLOCK_SIZE - 1) / BLOCK_SIZE);
                        kernDownSweep <<<fullBlocksPerGrid, BLOCK_SIZE >>> (nForScan, num, offset, devFalse);
                        offset /= 2;
                    }
                }
                
                int totalFalse;
                cudaMemcpy(&totalFalse, devFalse + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
                //std::cout << totalFalse << std::endl;
                if ((inp[n - 1] & (1 << bit)) != 0) totalFalse += 1;
                kernComputeIndices <<<fullBlocksPerGrid, BLOCK_SIZE>>> (n, totalFalse, devFalse, devTrue, devIndices);

                kernRadixSortScatter <<<fullBlocksPerGrid, BLOCK_SIZE>>> (n, devIndices, devInp, devTrue); //temporarily store output into devTrue buffer
                //cudaMemcpy(out, devTrue, n * sizeof(int), cudaMemcpyDeviceToHost);
                //for (int i = 0; i < 64; i++) {
                //    std::cout << out[i] << " ";
                //}
                //std::cout << std::endl;
                std::swap(devInp, devTrue);
            }

            if (enableTimer) timer().endGpuTimer();

            cudaMemcpy(out, devInp, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy devInp failed!");
            cudaFree(devInp);
            checkCUDAError("cudaFree devInp failed!");
            cudaFree(devTrue);
            checkCUDAError("cudaFree devTrue failed!");
            cudaFree(devFalse);
            checkCUDAError("cudaFree devFalse failed!");
            cudaFree(devIndices);
            checkCUDAError("cudaFree devIndices failed!");
        }
    }
}
