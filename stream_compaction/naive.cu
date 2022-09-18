#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"
#include <device_launch_parameters.h>

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        constexpr unsigned blockSize = 128; // TODO test different blockSizes

        __global__ void kernPrefixSumExclusiveScan(int d, int n, int *idata, int *odata) {
            unsigned index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index >= n) { return; }

            int odataidx = index; // clusmy but forces exclusive scan behavior
            if (d == 1) {
                if (++odataidx >= n) { return; }
            }

            unsigned cutoff = 1 << d - 1;
            odata[odataidx] = idata[index];
            if (index >= cutoff) {
                odata[odataidx] += idata[index - cutoff];
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            int* dev_data1;
            int* dev_data2;
            cudaMalloc((void**)&dev_data1, n * sizeof(int));
            cudaMalloc((void**)&dev_data2, n * sizeof(int));
            cudaMemcpy(dev_data1, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);

            timer().startGpuTimer();
            int d;
            for (d = 1; d <= ilog2ceil(n); ++d) {
                kernPrefixSumExclusiveScan<<<fullBlocksPerGrid, blockSize>>>(d, n, dev_data1, dev_data2);
                std::swap(dev_data1, dev_data2); // swap i/o arrays for next summing
                cudaDeviceSynchronize();
            }
            timer().endGpuTimer();

            // ensure we send back the last output bufer
            d % 2 == 0 ? 
                cudaMemcpy(odata, dev_data1, n * sizeof(int), cudaMemcpyDeviceToHost) :
                cudaMemcpy(odata, dev_data2, n * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(dev_data1);
            cudaFree(dev_data2);
        }
    }
}
