#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernUpStreamReduction(int n, int *odata, const int *idata) {
            
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
            // TODO
            int* dev_buffer1;
            int* dev_buffer2;
            int* dev_backup;

            dim3 gridSize(32, 32);
            dim3 blockSize(32, 32);

            // Memory allocation
            cudaMalloc((void**)&dev_buffer1, sizeof(int) * n);
            checkCUDAError("cudaMalloc dev_idata failed!");
            cudaMalloc((void**)&dev_buffer2, sizeof(int) * n);
            checkCUDAError("cudaMalloc dev_odata failed!");
            cudaMalloc((void**)&dev_backup, sizeof(int) * n);
            checkCUDAError("cudaMalloc dev_odata failed!");

            cudaMemcpy(dev_buffer1, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
            cudaMemcpy(dev_backup, dev_buffer1, sizeof(int) * n, cudaMemcpyDeviceToDevice);

            kernUpStreamReduction << <gridSize, blockSize >> > (n, dev_buffer2, dev_buffer1);

            cudaFree(dev_buffer1);
            cudaFree(dev_buffer2);
            cudaFree(dev_backup);
            timer().endGpuTimer();
        }

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
            // TODO
            timer().endGpuTimer();
            return -1;
        }
    }
}
