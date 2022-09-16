#include <iostream>
#include <cuda.h>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

#define BLOCK_SIZE 128

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        // TODO: __global__

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */

        __global__ void kernScanStep(int n, int offset, int* inp, int* out) {
            int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (idx >= n) return;
            if (idx < (offset + 1)) out[idx] = inp[idx];
            else {
                int outValue = inp[idx - offset] + inp[idx];
                out[idx] = outValue;
            }
        }

        __global__ void kernScanFirstStep(int n, int* inp, int* out) {
            int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (idx >= n) return;
            else if (idx == 0) out[0] = 0;
            else if (idx == 1) out[1] = inp[0];
            else out[idx] = inp[idx - 2] + inp[idx - 1];
        }

        void scan(int n, int *odata, const int *idata, bool enableTimer) {
            int* devInp;
            int* devOut;
            cudaMalloc((void**)&devInp, n * sizeof(int));
            checkCUDAError("cudaMalloc devInp failed!");
            cudaMalloc((void**)&devOut, n * sizeof(int));
            checkCUDAError("cudaMalloc devOut failed!");
            cudaMemcpy(devInp, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy idata failed!");

            if (enableTimer) timer().startGpuTimer();
            dim3 fullBlocksPerGrid((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
            kernScanFirstStep <<<fullBlocksPerGrid, BLOCK_SIZE >>>(n, devInp, devOut);
            //cudaMemcpy(odata, devOut, n * sizeof(int), cudaMemcpyDeviceToHost);
            //std::cout << "d0: ";
            //for (int i = 0; i < 32; i++) {
            //    std::cout << odata[i] << " ";
            //}
            //std::cout << std::endl;
            std::swap(devInp, devOut);
            for (int d = 1; d < ilog2ceil(n); d++) {
                //launch n-1-2^d threads; offset = 2^d; idxOffset = 2^d+1;
                int pow2d = pow(2, d);
                //fullBlocksPerGrid = dim3((n-1-pow2d + BLOCK_SIZE - 1) / BLOCK_SIZE);
                kernScanStep<<<fullBlocksPerGrid, BLOCK_SIZE>>>(n, pow2d, devInp, devOut);
                //cudaMemcpy(odata, devOut, n * sizeof(int), cudaMemcpyDeviceToHost);
                //std::cout << "d" << d << ": ";
                //for (int i = 0; i < 32; i++) {
                //    std::cout << odata[i] << " ";
                //}
                //std::cout << std::endl;
                std::swap(devInp, devOut);
            }
            if (enableTimer) timer().endGpuTimer();

            cudaMemcpy(odata, devInp, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy odata failed!");
            cudaFree(devInp);
            checkCUDAError("cudaFree devInp failed!");
            cudaFree(devOut);
            checkCUDAError("cudaFree devOut failed!");
        }
    }
}
