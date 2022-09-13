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
        __device__ void switchArray(int* odata, int* idata) {
            std::swap(odata, idata);
            return;
        }
        __global__ void kernScan(int n, int depth, int* odata, int* odata2, const int* idata) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index < pow(2, depth) || index >= n) {
                return;
            }

            odata2[index] = odata[index] + odata[index - int(pow(2, depth - 1))];
            switchArray(odata, odata2);
            return;
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
            // TODO
            int depth = ilog2ceil(n);
            int* odataReplacement;
            int oddEvenCount = 0;
            for (int i = 0; i < depth; i++) {
                dim3 blockDim((n + blockSize - 1) / blockSize);
                kernScan(n, i, odata, odataReplacement, idata);
                oddEvenCount += (n - pow(2, depth));
            }
            if (oddEvenCount % 2 != 0) {
                std::swap(odata, odataReplacement);
            }

            timer().endGpuTimer();
        }
    }
}
