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
        __global__ void kernScan(int n, int depth, int* odata, const int* idata) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n) {
                return;
            }
            int index1 = 1 << depth;
            //int index1 = 1 << depth;
            odata[index] = idata[index];
            if (index >= index1) {
                odata[index] += idata[index - index1];
            }
            return;
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            dim3 blockDim((n + blockSize - 1) / blockSize);
            int depth = ilog2ceil(n);
            int* input, *output;
            cudaMalloc((void**)&input, n*sizeof(int));
            cudaMalloc((void**)&output, n*sizeof(int));
            cudaMemcpy(input, idata, n*sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(output, idata, n * sizeof(int), cudaMemcpyHostToDevice);


            timer().startGpuTimer();
            for (int i = 0; i <= depth; ++i) {
                kernScan<<<blockDim, blockSize>>>(n, i, output, input);
                checkCUDAError("kernScan failed");
                //std::swap(input, output);
                cudaMemcpy(input, output, n * sizeof(int), cudaMemcpyDeviceToDevice);
            }

            timer().endGpuTimer();

            //cudaMemcpy(odata, output, n, cudaMemcpyDeviceToHost);
            //change from inclusive to excluvise
            cudaMemcpy(odata + 1, output, (n - 1)*sizeof(int), cudaMemcpyDeviceToHost);
            odata[0] = 0;
            cudaFree(input);
            cudaFree(output);
        }
    }
}
