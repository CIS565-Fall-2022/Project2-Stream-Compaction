#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

#define blockSize 256

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        __global__ void kernUpSweep(int n, int* data, int offset)
        {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n) {
                return;
            }

            if (index % (2 * offset) == 0) {
                int desIdx = index + (2 * offset) - 1;
                int srcIdx = index + offset - 1;

                data[desIdx] += data[srcIdx];
            }
        }

        __global__ void kernDownSweep(int n, int* data, int offset)
        {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n) {
                return;
            }
            if (index % (2 * offset) == 0) {
                int t = data[index + offset - 1];
                data[index + offset - 1] = data[index + offset * 2 - 1];
                data[index + offset * 2 - 1] += t;
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) 
        {
            int maxDepth = ilog2ceil(n);
            int maxSize = pow(2, maxDepth);
            dim3 fullBlocksPerGrid((maxSize + blockSize - 1) / blockSize);
            
            int *dev_data;
            cudaMalloc((void**)&dev_data, maxSize * sizeof(int));
            cudaMemcpy(dev_data, idata, maxSize * sizeof(int), cudaMemcpyHostToDevice);

            timer().startGpuTimer();
            // TODO
            // UpSweep
            for (int d = 0; d < maxDepth; d++) {
                kernUpSweep << < fullBlocksPerGrid, blockSize >> > (maxSize, dev_data, pow(2, d));
            }

            cudaMemset(dev_data + maxSize - 1, 0, sizeof(int));

            // DownSweep
            for (int d = maxDepth - 1; d >= 0; d--) {
                kernDownSweep << < fullBlocksPerGrid, blockSize >> > (maxSize, dev_data, pow(2, d));
            }

            timer().endGpuTimer();

            cudaMemcpy(odata, dev_data, n * sizeof(int), cudaMemcpyDeviceToHost);

            // free cuda memory
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
        int compact(int n, int *odata, const int *idata) 
        {
            int *dev_idata, *dev_odata, *dev_bool, *dev_idx;

            int maxDepth = ilog2ceil(n);
            int maxSize = pow(2, maxDepth);

            dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);
            dim3 maxBlocksPerGrid((maxSize + blockSize - 1) / blockSize);

            cudaMalloc((void**)&dev_idata, maxSize * sizeof(int));
            cudaMalloc((void**)&dev_odata, maxSize * sizeof(int));
            cudaMalloc((void**)&dev_bool, maxSize * sizeof(int));
            cudaMalloc((void**)&dev_idx, maxSize * sizeof(int));

            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            timer().startGpuTimer();
            // TODO

            Common::kernMapToBoolean << <fullBlocksPerGrid, blockSize >> > (n, dev_bool, dev_idata);
            cudaMemcpy(dev_idx, dev_bool, maxSize * sizeof(int), cudaMemcpyDeviceToDevice);

            // Scan
            // UpSweep
            for (int d = 0; d <= maxDepth - 1; d++) {
                kernUpSweep << < maxBlocksPerGrid, blockSize >> > (maxSize, dev_idx, pow(2, d));
            }

            cudaMemset(dev_idx + maxSize - 1, 0, sizeof(int));

            // DownSweep
            for (int d = maxDepth - 1; d >= 0; d--) {
                kernDownSweep << < maxBlocksPerGrid, blockSize >> > (maxSize, dev_idx, pow(2, d));
            }

            // Scatter
            //scatter
            Common::kernScatter << < fullBlocksPerGrid, blockSize >> > (maxSize, dev_odata, dev_idata, dev_bool, dev_idx);


            timer().endGpuTimer();

            cudaMemcpy(odata, dev_odata, sizeof(int) * n, cudaMemcpyDeviceToHost);

            // compute num of non-zero element
            int* arr = new int[maxSize];
            cudaMemcpy(arr, dev_bool, sizeof(int) * maxSize, cudaMemcpyDeviceToHost);

            int count = 0;
            for (int i = 0; i < maxSize; i++) {
                if (arr[i] == 1) {
                    count++;
                }
            }
            // Free cuda memory
            cudaFree(dev_idata);
            cudaFree(dev_odata);
            cudaFree(dev_bool);
            cudaFree(dev_idx);
            
            return count;
        }
    }
}
