#define GLM_FORCE_CUDA
#include <cuda.h>
#include <stdio.h>
#include <cmath>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_functions.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;

        #define blockSize 8

        int* dev_idata;
        int* dev_odata;

        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        // TODO: __global__

        __global__ void kernScan(int N, const int* idata, int* odata, int depth) {
            int k = threadIdx.x + (blockIdx.x * blockDim.x);
            if (k >= N) {
                return;
            }
            if (k >= 2^(depth-1)) {
                odata[k-1] = idata[k - 2 ^ (depth - 1)] + idata[k-1];
            }

            //kernScan(int N, odata, new int[], depth++)
            //Define a base case

            //cudaDeviceSynchronize();
            
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startGpuTimer();

            dim3 threadsPerBlock(n/blockSize);
            // TODO
            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            cudaMalloc((void**)&dev_odata, n * sizeof(int));

            cudaMemcpy(dev_idata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
            cudaMemcpy(dev_odata, odata, sizeof(int) * n, cudaMemcpyHostToDevice);

            kernScan <<<threadsPerBlock, blockSize >>> (n, dev_idata, dev_odata, 1);
            
            cudaMemcpy((void**)idata, dev_idata, sizeof(int) * n, cudaMemcpyDeviceToHost);
            cudaMemcpy(odata, dev_odata, sizeof(int) * n, cudaMemcpyDeviceToHost);

            cudaFree(dev_idata);
            cudaFree(dev_odata);

            timer().endGpuTimer();
        }
    }
}
