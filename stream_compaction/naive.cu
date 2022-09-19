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
        __global__ void kernScan(int n,  int depth, int offset, int* odata, const int* idata)
        {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n)
            {
                return;
            }
            if (index == 0)
            {
                odata[index] = 0;
            }
            if (index >= offset)
            {
                odata[index] = idata[index - offset] + idata[index];
            }
            else
            {
                odata[index] = idata[index];
            }  

        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            int blockSize = 128;
            int numBlocks = ((n + blockSize - 1) / blockSize);
            int* dev_outDataA;
            int* dev_outDataB;
            cudaMalloc((void**)&dev_outDataA, n * sizeof(int));
            checkCUDAError("Malloc dev_outDataA Failed! ");
            cudaMalloc((void**)&dev_outDataB, n * sizeof(int));
            checkCUDAError("Malloc dev_outDataB Failed! ");
            //cudaMemset(dev_outDataA, 0, n * sizeof(int));
            checkCUDAError("Memset dev_outDataA Failed! ");
            cudaDeviceSynchronize();
            cudaMemcpy(dev_outDataA, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
            checkCUDAError("Memcpy dev_outDataA Failed! ");
            cudaDeviceSynchronize();
            timer().startGpuTimer();
            int d = ilog2ceil(n);
            for (int k = 1; k <= d; k ++)
            {
                int offset = 1 << (k - 1);
                kernScan <<< numBlocks, blockSize >>> (n, k, offset, dev_outDataB, dev_outDataA);
                std::swap(dev_outDataB, dev_outDataA);
            }
            timer().endGpuTimer();
            cudaMemcpy(odata, dev_outDataA, sizeof(int) * n, cudaMemcpyDeviceToHost);
            checkCUDAError("Memcpy Back dev_outDataA Failed! ");
            //shift
            for (int i = n - 1; i > 0; i--)
            {
                odata[i] = odata[i - 1];
            }
            odata[0] = 0;
            cudaFree(dev_outDataA);
            cudaFree(dev_outDataB);

        }
    }
}