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

        void scan(int n, int *odata, const int *idata, bool enableTimer) {
            int* devInp;
            cudaMalloc((void**)&devInp, n * sizeof(int));
            checkCUDAError("cudaMalloc devInp failed!");
            cudaMemcpy(devInp, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy idata failed!");

            if (enableTimer) timer().startGpuTimer();

            // up sweep
            int num = n / 2;
            for (int d = 0; d < ilog2ceil(n); d++) {
                int offset = 1 << d;
                dim3 fullBlocksPerGrid((num + BLOCK_SIZE - 1) / BLOCK_SIZE);
                kernUpSweep<<<fullBlocksPerGrid, BLOCK_SIZE>>>(n, num, offset, devInp);
                num /= 2;
            }
            cudaMemset(devInp+n-1, 0, sizeof(int));

            // down sweep
            int offset = n / 2;
            for (int d = 0; d < ilog2ceil(n); d++) {
                int num = 1 << d;
                dim3 fullBlocksPerGrid((num + BLOCK_SIZE - 1) / BLOCK_SIZE);
                kernDownSweep<<<fullBlocksPerGrid, BLOCK_SIZE>>> (n, num, offset, devInp);
                offset /= 2;
            }

            if (enableTimer) timer().endGpuTimer();

            cudaMemcpy(odata, devInp, n * sizeof(int), cudaMemcpyDeviceToHost);
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


        int compact(int n, int *odata, const int *idata, bool enableTimer) {
            int* devInp;
            int* devBools;
            int* devOut;
            cudaMalloc((void**)&devInp, n * sizeof(int));
            checkCUDAError("cudaMalloc devInp failed!");
            cudaMalloc((void**)&devOut, n * sizeof(int));
            checkCUDAError("cudaMalloc devOut failed!");
            cudaMalloc((void**)&devBools, n * sizeof(int));
            checkCUDAError("cudaMalloc devBools failed!");
            cudaMemcpy(devInp, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy idata failed!");

            if (enableTimer) timer().startGpuTimer();
            dim3 fullBlocksPerGrid((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
            Common::kernMapToBoolean<<<fullBlocksPerGrid, BLOCK_SIZE>>> (n, devBools, devInp);

            // up sweep
            int num = n / 2;
            for (int d = 0; d < ilog2ceil(n); d++) {
                int offset = 1 << d;
                dim3 fullBlocksPerGrid((num + BLOCK_SIZE - 1) / BLOCK_SIZE);
                kernUpSweep << <fullBlocksPerGrid, BLOCK_SIZE >> > (n, num, offset, devBools);
                num /= 2;
            }
            cudaMemset(devBools + n - 1, 0, sizeof(int));

            // down sweep
            int offset = n / 2;
            for (int d = 0; d < ilog2ceil(n); d++) {
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
            cudaMemcpy(indices.get(), devBools, n * sizeof(int), cudaMemcpyDeviceToHost);
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
            return idata[n - 1] != 0 ? indices[n - 1] + 1 : indices[n - 1];
            return -1;
        }
    }
}
