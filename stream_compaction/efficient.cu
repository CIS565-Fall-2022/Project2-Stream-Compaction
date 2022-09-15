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
            dim3 fullBlocksPerGrid((blockSize + intermArraySize - 1) / blockSize);


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

        void scanRecursion(int n, int* data, dim3 blockPerGrid)
        {
            for (int d = 0; d <= ilog2ceil(n) - 1; ++d) {
                kernUpSweep << < blockPerGrid, blockSize >> > (n, d, data);
            }

            cudaMemset(data + n - 1, 0, sizeof(int));
            checkCUDAError("cudaMemset failed!");


            for (int d = ilog2ceil(n) - 1; d >= 0; --d) {
                kernDownSweep << < blockPerGrid, blockSize >> > (n, d, data);
            }

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
            dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);

            int* dev_data;
            int* dev_bool;
            cudaMalloc((void**)&dev_data, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_data failed!");
            cudaMalloc((void**)&dev_bool, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_data failed!");

            timer().startGpuTimer();
            // Step 1
            StreamCompaction::Common::kernMapToBoolean << <fullBlocksPerGrid, blockSize >> > (n, dev_bool, idata);
            cudaDeviceSynchronize();
            
            cudaMemcpy(dev_data, dev_bool, sizeof(int) * n, cudaMemcpyDeviceToDevice);

            // Step 2
            scanRecursion(1 << ilog2ceil(n), dev_data, fullBlocksPerGrid);

            // Step 3
             StreamCompaction::Common::kernScatter <<<fullBlocksPerGrid, blockSize >>>(n, odata,
                idata, dev_bool, dev_data);


            timer().endGpuTimer();
            int returnSize = 0;
            cudaMemcpy(&returnSize, dev_data + n - 1, sizeof(int), cudaMemcpyDeviceToHost);


            cudaFree(dev_data);
            cudaFree(dev_bool);

            return returnSize;
        }
    }
}
