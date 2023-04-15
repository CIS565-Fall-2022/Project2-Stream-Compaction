#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

const int blockSize = 128;

__device__ inline int twoPow(int d) {
    return (1 << (d));
}

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        // TODO: __global__

        __global__ void kernNaiveScan(int n, int d, int *odata, int *idata) {
            int idx = blockDim.x * blockIdx.x + threadIdx.x;
            if (idx >= n) return;
            // Add adjacent elements to get the prefix sum
            if (idx >= twoPow(d - 1)) 
                odata[idx] = idata[idx - twoPow(d - 1)] + idata[idx];
            else
                odata[idx] = idata[idx];
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            int *dev_idata;
            int *dev_odata;
            dim3 blocksPerGrid((n + blockSize - 1) / blockSize);
            // allocate
            cudaMalloc((void **)&dev_idata, n * sizeof(int));
            checkCUDAError("allcoate dev_idata failed!\n");
            cudaMalloc((void **)&dev_odata, n * sizeof(int));
            checkCUDAError("allcoate dev_odata failed!\n");

            // move data to device
            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            
            timer().startGpuTimer();
            // TODO: Naive Scan
            for (int d = 1; d <= ilog2ceil(n); d++) {
                kernNaiveScan<<<blocksPerGrid, blockSize>>>(n, d, dev_odata, dev_idata);
                std::swap(dev_odata, dev_idata);
            }
            timer().endGpuTimer();

            // shift right
            odata[0] = 0;
            cudaMemcpy(odata + 1, dev_idata, (n - 1) * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(dev_idata);
            cudaFree(dev_odata);
        }
    }
}
