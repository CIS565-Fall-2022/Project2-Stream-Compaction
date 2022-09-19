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

        __global__ void kernUpSweep(int n, int t, int* data) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index < n) {
                data[t * index + t - 1] += data[t * index + (t >> 1) - 1];
            }
        }
        __global__ void kernDownSweep(int n, int t, int* data) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index < n) {
                int tmp = data[t * index + (t >> 1) - 1];
                data[t * index + (t >> 1) - 1] = data[t * index + t - 1];
                data[t * index + t - 1] += tmp;
            }
        }
        __global__ void kernSetArray(int value, int index, int* data) {
            data[index] = value;
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            int blockSize = 128;
            int* buffer;
            int N = 1 << ilog2ceil(n);
            cudaMalloc((void**)&buffer, N * sizeof(int));
            cudaMemcpy(buffer, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            timer().startGpuTimer();
            for (int d = 0; d < ilog2ceil(n); d++) {
                int computeCount = n >> d + 1;  //up sweep is n, down sweep is N. It's not a typo
                dim3 blockCount((computeCount + blockSize - 1) / blockSize);
                kernUpSweep << <blockCount, blockSize >> > (computeCount, 1<<d+1, buffer);  //todo non power of 2
            }
            //cudaMemset(buffer + N - 1, 0, sizeof(int)); it cost to much
            kernSetArray << <1, 1 >> > (0, N - 1, buffer);
            for (int d = ilog2ceil(n) - 1; d >= 0; d--) {
                int computeCount = N >> d + 1;
                dim3 blockCount((computeCount + blockSize - 1) / blockSize);
                kernDownSweep << <blockCount, blockSize >> > (computeCount, 1 << d+1, buffer);
            }
            timer().endGpuTimer();

            cudaMemcpy(odata, buffer, n * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(buffer);
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
            int* bools, * indices, * ibuffer, * obuffer;
            cudaMalloc((void**)&bools, n * sizeof(int));
            cudaMalloc((void**)&indices, n * sizeof(int));
            cudaMalloc((void**)&ibuffer, n * sizeof(int));
            cudaMalloc((void**)&obuffer, n * sizeof(int));
            cudaMemcpy(ibuffer, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            int blockSize = 128;
            dim3 blockCount((n + blockSize - 1) / blockSize);

            timer().startGpuTimer();
            Common::kernMapToBoolean<<<blockCount, blockSize>>>(n, bools, ibuffer);

            //------------------ scan ---------------------
            int N = 1 << ilog2ceil(n);
            cudaMemcpy(indices, bools, n * sizeof(int), cudaMemcpyDeviceToDevice);

            for (int d = 0; d < ilog2ceil(n); d++) {
                int computeCount = n >> d + 1;  //up sweep is n, down sweep is N. It's not a typo
                dim3 blockCount((computeCount + blockSize - 1) / blockSize);
                kernUpSweep << <blockCount, blockSize >> > (computeCount, 1 << d + 1, indices);  //todo non power of 2
            }
            kernSetArray << <1, 1 >> > (0, N - 1, indices);
            for (int d = ilog2ceil(n) - 1; d >= 0; d--) {
                int computeCount = N >> d + 1;
                dim3 blockCount((computeCount + blockSize - 1) / blockSize);
                kernDownSweep << <blockCount, blockSize >> > (computeCount, 1 << d + 1, indices);
            }
            //------------------ scan ---------------------

            Common::kernScatter << <blockCount, blockSize >> > (n, obuffer, ibuffer, bools, indices);
            timer().endGpuTimer();

            int count, bias;
            cudaMemcpy(&count, indices + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&bias, bools + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            int remainNum = count + bias;
            cudaMemcpy(odata, obuffer, remainNum * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(bools);
            cudaFree(indices);
            cudaFree(ibuffer);
            cudaFree(obuffer);
            return remainNum;   //todo
        }
    }
}
