#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"
#include "math_functions.h"
#include "device_functions.h"

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        void printArray(int n, int* a, bool abridged = false) {
            printf("    [ ");
            for (int i = 0; i < n; i++) {
                if (abridged && i + 2 == 15 && n > 16) {
                    i = n - 2;
                    printf("... ");
                }
                printf("%3d ", a[i]);
            }
            printf("]\n");
        }
        
        __global__ void kernNaiveScan(int d, int N, int* odata, const int* idata) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;

            if (index > N) {
                return;
            }

            if (index >= (int)(pow(2, d - 1) + .1)) {
                odata[index] = idata[index - (int)(pow(2, d - 1) + .1)] + idata[index];
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
            int* dev_in;
            int* dev_out;
            cudaMalloc((void**)&dev_in, n * sizeof(int));
            cudaMalloc((void**)&dev_out, n * sizeof(int));
            cudaMemcpy(dev_in, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
            //cudaMemcpy(dev_out, odata, sizeof(int) * n, cudaMemcpyHostToDevice);

            //cudaMemcpy(odata, dev_in, sizeof(int) * n, cudaMemcpyDeviceToHost);
            //printArray(n, odata, true);

            timer().startGpuTimer();
            int blockSize = 128;
            dim3 fullBlocks((n + blockSize - 1) / blockSize);
            for (int d = 1; d <= ilog2ceil(n); d++) {
                kernNaiveScan << <fullBlocks, blockSize >> > (d, n, dev_out, dev_in);
                int* temp = dev_out;
                dev_out = dev_in;
                dev_in = temp;

                /*cudaMemcpy(odata, dev_in, sizeof(int) * n, cudaMemcpyDeviceToHost);
                printArray(n, odata, true);*/
            }
            timer().endGpuTimer();
   
            cudaMemcpy(odata + 1, dev_in, sizeof(int) * (n - 1), cudaMemcpyDeviceToHost);
            cudaFree(dev_in);
            cudaFree(dev_out);
            odata[0] = 0;          
        }
    }
}
