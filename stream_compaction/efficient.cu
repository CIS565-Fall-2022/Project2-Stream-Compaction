#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"
#include <device_launch_parameters.h>
#define blockSize 128
namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernUpStreamReduction(int n, int d, int* data) {
            int k = threadIdx.x + (blockIdx.x * blockDim.x);
            int offsetd1 = pow(2, d + 1);
            int offsetd = pow(2, d);
            if (k >= n) {
                return;
            }
            /*if (k % offsetd1 != 0) {
                return;
            }*/
            if (k % offsetd1 == 0) {
                data[k + offsetd1 - 1] = data[k + offsetd1 - 1] + data[k + offsetd - 1];
                return;
            }
        }

        __global__ void kernDownStream(int n, int d, int* data) {
            int k = threadIdx.x + (blockIdx.x * blockDim.x);
            int offsetd1 = pow(2, d + 1);
            int offsetd = pow(2, d);
            if (k >= n) {
                return;
            }
            if (k % offsetd1 != 0) {
                return;
            }
            int t = data[k - 1 + offsetd];               // Save left child
            data[k - 1 + offsetd] = data[k - 1 + offsetd1];  // Set left child to this node’s value
            data[k - 1 + offsetd1] += t;
        }


        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int* odata, const int* idata) {
            timer().startGpuTimer();
            // TODO
            int* dev_data;

            //dim3 gridSize(32, 32);
            //dim3 blockSize(32, 32);

            dim3 blocksPerGrid((n + blockSize - 1) / blockSize);

            int maxDepth = ilog2ceil(n);
            int extended_n = pow(2, maxDepth);
            int* extended_idata = new int[extended_n];

            for (int i = 0; i < extended_n; i++) {
                extended_idata[i] = (i < n) ? idata[i] : 0;
            }
            // Memory allocation
            cudaMalloc((void**)&dev_data, sizeof(int) * extended_n);
            checkCUDAError("cudaMalloc dev_data failed!");

            cudaMemcpy(dev_data, extended_idata, sizeof(int) * extended_n, cudaMemcpyHostToDevice);
            checkCUDAError("memcpy into dev_data failed!");

            for (int d = 0; d < maxDepth; d++) {    // where d is depth of iteration
                kernUpStreamReduction << <blocksPerGrid, blockSize >> > (extended_n, d, dev_data);
                checkCUDAError("kernUpStreamReduction invocation failed!");
            }

            int* lastVal = new int();
            cudaMemcpy(lastVal, dev_data + extended_n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("lastVal memcpy failed!");

            cudaMemset(dev_data + extended_n - 1, 0, sizeof(int));
            for (int d = maxDepth - 1; d >= 0; d--) {    // where d is depth of iteration
                kernDownStream << <blocksPerGrid, blockSize >> > (extended_n, d, dev_data);
                checkCUDAError("kernDownStream invocation failed!");
            }

            cudaMemcpy(odata, dev_data, sizeof(int) * (extended_n), cudaMemcpyDeviceToHost);
            checkCUDAError("odata memcpy failed!");

            cudaFree(dev_data);
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
        int compact(int n, int* odata, const int* idata) {
            timer().startGpuTimer();
            // TODO
            timer().endGpuTimer();
            return -1;
        }
    }
}
