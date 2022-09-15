 #include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

#define blockSize 256
dim3 threadPerBlock(blockSize);

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        // TODO: __global__
        __global__ void kernNaiveScan(int n, int* idata, int* odata, int offset) {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index > n) {
                return;
            }
            if (index >= offset) {
                odata[index] = idata[index - offset] + idata[index];
            }
            else {
                odata[index] = idata[index];
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            
            // TODO
            dim3 blockPerGrid((n + blockSize - 1) / blockSize);

            int* dev_buf1;
            int* dev_buf2;
            // create memory
            cudaMalloc((void**)&dev_buf1, n * sizeof(int));
            cudaMalloc((void**)&dev_buf2, n * sizeof(int));
            // copy data
            cudaMemcpy(dev_buf1, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            timer().startGpuTimer();
            for (int i = 1; i <= ilog2ceil(n); ++i) {
                int offset = pow(2, i - 1);
                kernNaiveScan << <blockPerGrid, threadPerBlock >> > (n, dev_buf1, dev_buf2, offset);
                cudaMemcpy(dev_buf1, dev_buf2, n * sizeof(int), cudaMemcpyDeviceToDevice);
            }
            timer().endGpuTimer();
            odata[0] = 0;
            cudaMemcpy(odata + 1, dev_buf1, n * sizeof(int), cudaMemcpyDeviceToHost);
            
            // free memory
            cudaFree(dev_buf1);
            cudaFree(dev_buf2);
        }
    }
}
