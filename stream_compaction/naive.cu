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
        __global__ void KernNaiveScanIteration(int n, int d, int* odata, int* idata)
        {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n )
            {
                return;
            }
            if (index >= (1 << (d - 1)))
                odata[index] = idata[index - (1 << (d - 1))] + idata[index];
            else
                odata[index] = idata[index];
        }


        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);

            int* dev_odata1;
            int* dev_odata2;
            cudaMalloc((void**)&dev_odata1, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_odata1 failed!");
            cudaMalloc((void**)&dev_odata2, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_odata2 failed!");

            cudaMemcpy(dev_odata1, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
            cudaMemcpy(dev_odata2, idata, sizeof(int) * n, cudaMemcpyHostToDevice);

            timer().startGpuTimer();
            
            int nParity = ilog2ceil(n) % 2;

            for (int d = 1; d <= ilog2ceil(n); ++d)
            {
                if (d % 2 != nParity)
                {
                    KernNaiveScanIteration << <fullBlocksPerGrid, blockSize >> > (n, d, dev_odata2, dev_odata1);
                }
                else
                {
                    KernNaiveScanIteration << <fullBlocksPerGrid, blockSize >> > (n, d, dev_odata1, dev_odata2);
                }
            }

            // Inclusive to exclusive
            odata[0] = 0;
            cudaMemcpy(odata + 1, dev_odata1, sizeof(int) * (n - 1), cudaMemcpyDeviceToHost);
            timer().endGpuTimer();

            //Clean up
            cudaFree(dev_odata1);
            cudaFree(dev_odata2);

        }
    }
}
