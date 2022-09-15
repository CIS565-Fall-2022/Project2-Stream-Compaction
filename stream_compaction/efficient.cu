#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

#define blockSize 128

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernUpSweep(int n, int d, int* x)
        {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n) {
                return;
            }
            if (index % (1 << (d+1)) == 0) 
            {

                x[index + (1 << (d + 1)) - 1] += x[index + (1 << d ) - 1];
            }
        }

        __global__ void kernDownSweep(int n, int d, int* x)
        {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n) {
                return;
            }

            if (index % (1 << (d + 1)) == 0)
            {

                int t = x[index + (1 << d) - 1];
                x[index + (1 << d) - 1] = x[index + (1 << (d + 1)) - 1];
                x[index + (1 << (d + 1)) - 1] += t;
            }
        }


        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int* odata, const int* idata) {


            int intermArraySize = 1 << ilog2ceil(n);
            dim3 fullBlocksPerGrid((n + intermArraySize - 1) / blockSize);


            int* dev_data;
            cudaMalloc((void**)&dev_data, intermArraySize * sizeof(int));
            checkCUDAError("cudaMalloc dev_data failed!");
            cudaMemcpy(dev_data, idata, sizeof(int) * n, cudaMemcpyHostToDevice);

            timer().startGpuTimer();
            for (int d = 0; d <= ilog2ceil(intermArraySize) - 1; ++d) {
                kernUpSweep << < fullBlocksPerGrid, blockSize >> > (intermArraySize, d, dev_data);
                //cudaDeviceSynchronize();
            }

            cudaMemset(dev_data + intermArraySize - 1, 0, sizeof(int));
            checkCUDAError("cudaMemset failed!");


            for (int d = ilog2ceil(intermArraySize) - 1; d >= 0; --d) {
                kernDownSweep << < fullBlocksPerGrid, blockSize >> > (intermArraySize, d, dev_data);
                //cudaDeviceSynchronize();
            }


            timer().endGpuTimer();
            cudaMemcpy(odata, dev_data, sizeof(int) * n, cudaMemcpyDeviceToHost);
            cudaFree(dev_data);

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
        int compact(int n, int* odata, const int* idata) {
            timer().startGpuTimer();
            // TODO
            timer().endGpuTimer();
            return -1;
        }
    }
}
