#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "common.h"
#include "naive.h"
#include <device_launch_parameters.h>
#include <device_functions.h>
#include <iostream>

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        // TODO: __global__
        
        __global__ void kernScanStep(int n, int stride, int* idata, int* odata) {
            int idx = threadIdx.x + (blockIdx.x * blockDim.x);
            if (idx >= n) {
                return;
            }
            odata[idx] = idata[idx];
            if (idx < stride) {
                return;
            }
            odata[idx] += idata[idx - stride];
        }
        
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);

            int* dev_idata, * dev_odata;

            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            checkCUDAError("Error during cudaMalloc dev_idata");

            cudaMalloc((void**)&dev_odata, n * sizeof(int));
            checkCUDAError("Error during cudaMalloc dev_odata");

            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("Error during cudaMemcpy dev_idata");
            
            cudaDeviceSynchronize();

            // TODO

            timer().startGpuTimer();

            for (int d = 0; d <= ilog2ceil(n); d++) { 
                kernScanStep << <fullBlocksPerGrid, blockSize >> > (n, std::pow(2, d), dev_idata, dev_odata);
                cudaDeviceSynchronize();

                cudaMemcpy(dev_idata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToDevice);
                checkCUDAError("Error during cudaMemcpy dev_odata ==> dev_idata");
            }

            timer().endGpuTimer();

            cudaMemcpy(odata + 1, dev_odata, (n - 1) * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("Error during cudaMemcpy odata");
            
            cudaFree(dev_idata);
            checkCUDAError("Error during cudaFree dev_idata");
            cudaFree(dev_odata);
            checkCUDAError("Error during cudaFree dev_odata");


            cudaDeviceSynchronize();
        }

    }
}
