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
        __global__ void kernNaiveScan(int n, int* odata, int* idata, int stride) 
        {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n) {
                return;
            }

            if (index >= stride) {
                odata[index] = idata[index - stride] + idata[index];
            }
            else {
                odata[index] = idata[index];
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
            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            timer().startGpuTimer();
            // TODO
            for (int d = 1; d <= ilog2ceil(n); d++) {
                kernNaiveScan << < fullBlocksPerGrid, blockSize >> > (n, dev_odata, dev_idata, pow(2.0,d-1));
                
                // ping-pong buffer
                int* tmp = dev_idata;
                dev_idata = dev_odata;
                dev_odata = tmp;
            }

            timer().endGpuTimer();

            // covert from inclusive scan to exclusive scan
            // copy the memory from the second index and manually set identity to the first element
            cudaMemcpy(odata + 1, dev_idata, (n-1) * sizeof(int), cudaMemcpyDeviceToHost);
            odata[0] = 0;

            // free cuda memory
            cudaFree(dev_odata);
            cudaFree(dev_idata);
        }
    }
}
