#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
    namespace Naive { 
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer() {
            static PerformanceTimer timer;
            return timer;
        }
        // TODO: __global__

        __global__ void kernPartialScan(int* out, int* in, int n, int stride) {
            int idx = blockDim.x * blockIdx.x + threadIdx.x;
            if (idx >= n) {
                return;
            }
            out[idx] = in[idx] + in[idx - stride];
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            // TODO
            size_t bytes = n * sizeof(int);
            int* devBuf, * devTmp;
            cudaMalloc(&devBuf, bytes);
            cudaMalloc(&devTmp, bytes);
            cudaMemcpy(devBuf, idata, bytes, cudaMemcpyKind::cudaMemcpyHostToDevice);

            timer().startGpuTimer();

            int stride = 1;
            while (stride < n) {
                int num = n - stride;
                int blockSize = Common::getDynamicBlockSizeEXT(num);
                int blockNum = ceilDiv(num, blockSize);
                kernPartialScan<<<blockNum, blockSize>>>(devTmp + stride, devBuf + stride, n - stride, stride);
                cudaMemcpy(devBuf + stride, devTmp + stride, num * sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToDevice);
                stride <<= 1;
            }
            timer().endGpuTimer();

            odata[0] = 0;
            cudaMemcpy(odata + 1, devBuf, (n - 1) * sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost);
            cudaFree(devBuf);
            cudaFree(devTmp);
        }
    }
}
