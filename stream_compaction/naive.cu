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
        // TODO: __global__
        __global__ void kernNaiveScan(int n, int stride, int *idata, int *odata)
        {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n) return;

            if (index >= stride)
            {
                odata[index] = idata[index - stride] + idata[index];
            }
            else
            {
                odata[index] = idata[index];
            }
        }

        __global__ void kernShiftRight(int n, int* idata, int* odata)
        {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n) return;

            if (index == 0)
            {
                odata[index] = 0;
            }
            else
            {
                odata[index] = idata[index - 1];
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            int *dev_idata;
            int *dev_odata;
            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            cudaMalloc((void**)&dev_odata, n * sizeof(int));
            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            timer().startGpuTimer();
            // TODO
            int blockSize = 128;
            dim3 blockNum = (n + blockSize - 1) / blockSize;
            for (int stride = 1; stride < 2 * n; stride *= 2)
            {
                kernNaiveScan<<<blockNum, blockSize>>>(n, stride, dev_idata, dev_odata);
                std::swap(dev_idata, dev_odata);

            }
            // Convert from inclusive scan to exclusive scan
            // No need to swap buffers here since they're swapped in for loop.
            kernShiftRight<<<blockNum, blockSize>>>(n, dev_idata, dev_odata);

            timer().endGpuTimer();

            cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(dev_idata);
            cudaFree(dev_odata);
        }
    }
}
