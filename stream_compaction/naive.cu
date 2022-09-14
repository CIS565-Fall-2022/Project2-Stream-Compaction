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
            timer().startGpuTimer();
            // TODO
            size_t bytes = n * sizeof(int);
            int* buf, * tmp;
            cudaMalloc(&buf, bytes);
            cudaMalloc(&tmp, bytes);
            cudaMemcpy(buf, idata, bytes, cudaMemcpyKind::cudaMemcpyHostToDevice);

            int stride = 1;
            while (stride < n) {
                int num = n - stride;
                int blockSize = Common::getDynamicBlockSizeEXT(num);
                int blockNum = (num + blockSize - 1) / blockSize;
                kernPartialScan<<<blockNum, blockSize>>>(tmp + stride, buf + stride, n - stride, stride);
                cudaMemcpy(buf + stride, tmp + stride, num * sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToDevice);
                stride <<= 1;
            }
            odata[0] = 0;
            cudaMemcpy(odata + 1, buf, (n - 1) * sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost);
            cudaFree(buf);
            cudaFree(tmp);

            timer().endGpuTimer();
        }
    }
}
