#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        // TODO: __global__
        __global__ void kernScan(int n, int bias, int* odata, const int* idata) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index >= bias && index < n) {
                odata[index] = idata[index - bias] + idata[index];
            }
            else {
                odata[index] = idata[index];
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int* odata, const int* idata) {
            int blockSize = 128;
            dim3 blockCount((n + blockSize - 1) / blockSize);

            int* buffer1, *buffer2;
            cudaMalloc((void**)&buffer1, n * sizeof(int));
            cudaMalloc((void**)&buffer2, n * sizeof(int));
            cudaMemcpy(buffer1, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            timer().startGpuTimer();
            int in = ilog2ceil(n);
            for (int d = 1; d <= in; d++) {
                int bias = 1 << d - 1;
                kernScan<<<blockCount, blockSize>>>(n, bias, buffer2, buffer1);

                std::swap(buffer1, buffer2);
            }
            timer().endGpuTimer();

            cudaMemcpy(odata + 1, buffer1, (n - 1) * sizeof(int), cudaMemcpyDeviceToHost);
            odata[0] = 0;
            cudaFree(buffer1);
            cudaFree(buffer2);
        }
    }
}
