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
        __global__ void kernScan(int n, int d, int* odata, const int* idata) {
            int k = threadIdx.x + (blockIdx.x * blockDim.x);
            if (k >= n) {
                return;
            }
            int stride = 1 << (d - 1);
            if (k >= stride) {
                odata[k] = idata[k - stride] + idata[k];
            }
            else {
                odata[k] = idata[k];
            }
        }
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */     
        void scan(int n, int *odata, const int *idata) {
            int* dev_idata;
            int* dev_odata;

            dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);

            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            cudaMalloc((void**)&dev_odata, n * sizeof(int));

            cudaMemcpy(dev_idata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
          //  cudaMemcpy(dev_odata, odata, sizeof(int) * n, cudaMemcpyHostToDevice);

            timer().startGpuTimer();
            // TODO
            for (int d = 1; d <= ilog2ceil(n); d++) {
                kernScan << <fullBlocksPerGrid, blockSize >> > (n, d, dev_odata, dev_idata);
                std::swap(dev_odata, dev_idata);
            }
            timer().endGpuTimer();
            //convert from inclusive scan to exclusive scan
            odata[0] = 0;
            cudaMemcpy(odata + 1, dev_idata, sizeof(int) * (n - 1), cudaMemcpyDeviceToHost);

            cudaFree(dev_idata);
            cudaFree(dev_odata);
            
        }
    }
}
