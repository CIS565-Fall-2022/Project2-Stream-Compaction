#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

#define blockSize 128

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        // TODO: __global__
        __global__ void kernNaive(int n, int check, const int* dev_idata, int* dev_odata) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index < n) {
                if (index >= check) {
                    dev_odata[index] = dev_idata[index - check] + dev_idata[index];
                }
                else {
                    dev_odata[index] = dev_idata[index];
                }
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            // TODO
            int* dev_idata;
            int* dev_odata;
            dim3 fullBlocksPerGrid((blockSize + n - 1) / blockSize);

            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            checkCUDAError("generate dev_temp1 failed!");

            cudaMalloc((void**)&dev_odata, n * sizeof(int));
            checkCUDAError("generate dev_temp2 failed!");

            //make dev_idata shift to right by 1
            int identity = 0;
            cudaMemcpy(dev_idata, &identity, sizeof(int), cudaMemcpyHostToDevice);

            cudaMemcpy(&dev_idata[1], idata, (n - 1) * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("memory copy to dev_idata failed!");
            
            timer().startGpuTimer();

            for (int d = 1; d <= ilog2ceil(n); ++d) {
                int check = pow(2, d - 1);
                kernNaive << <fullBlocksPerGrid, blockSize >> > (n, check, dev_idata, dev_odata);
                std::swap(dev_idata, dev_odata);
            }

            timer().endGpuTimer();

            cudaMemcpy(odata, dev_idata, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("memory copy to odata failed!");

            cudaFree(dev_idata);
            cudaFree(dev_odata);

        }
    }
}
