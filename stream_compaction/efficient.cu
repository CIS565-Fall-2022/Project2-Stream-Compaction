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

        __global__ void kernUpStreamReduction(int n, int d, int *data) {
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
        void scan(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
            // TODO
            int* dev_buffer1;
            int* dev_buffer2;
            int* dev_backup;

            //dim3 gridSize(32, 32);
            //dim3 blockSize(32, 32);

            dim3 blocksPerGrid((n + blockSize - 1) / blockSize);

            // Memory allocation
            cudaMalloc((void**)&dev_buffer1, sizeof(int) * n);
            checkCUDAError("cudaMalloc dev_idata failed!");
            cudaMalloc((void**)&dev_buffer2, sizeof(int) * n);
            checkCUDAError("cudaMalloc dev_odata failed!");
            cudaMalloc((void**)&dev_backup, sizeof(int) * n);
            checkCUDAError("cudaMalloc dev_odata failed!");

            for (int i = 0; i < n; i++) {
                printf("%d, ", idata[i]);
            }

            cudaMemcpy(dev_buffer1, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
            checkCUDAError("memcpy into  dev_buffer1 failed!");
            cudaMemcpy(dev_backup, dev_buffer1, sizeof(int) * n, cudaMemcpyDeviceToDevice);
            checkCUDAError("memcpy into dev_backup failed!");
            
            int maxDepth = ilog2ceil(n);
            for (int d = 0; d < maxDepth; d++) {    // where d is depth of iteration
                kernUpStreamReduction << <blocksPerGrid, blockSize >> > (n, d, dev_buffer1);
                checkCUDAError("kernUpStreamReduction invocation failed!");
            }

            int* lastVal = new int();
            cudaMemcpy(lastVal, dev_buffer1 + n - 1, sizeof(int), cudaMemcpyDeviceToHost);

            cudaMemset(dev_buffer1 + n - 1, 0, sizeof(int));
            for (int d = maxDepth - 1; d >= 0; d--) {    // where d is depth of iteration
                kernDownStream << <blocksPerGrid, blockSize >> > (n, d, dev_buffer1);
                checkCUDAError("kernDownStream invocation failed!");
            }
            
            cudaMemcpy(odata, dev_buffer1, sizeof(int) * (n), cudaMemcpyDeviceToHost);

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
