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

        //// DEBUGGER TEST
        //__global__ void kernTestDebugger(int param) {
        //    int index = threadIdx.x + (blockIdx.x * blockDim.x);
        //    index = 1;
        //    index = threadIdx.x + (blockIdx.x * blockDim.x);
        //    param = index;
        //}

        // TODO: __global__
        __global__ void kernNaiveScan(int n, int d, int *odata, const int *idata) {
            int k = threadIdx.x + blockIdx.x * blockDim.x;
            int offset = pow(2, d - 1);
            if (k >= n) {
                return;
            }
            
            if (k >= offset) {
                odata[k] = idata[k - offset] + idata[k];
            }
            else {
                odata[k] = idata[k];
            }

        }

        __global__ void kernInclusiveToExclusive(int n, int* odata, const int* idata) {
            // shift all elements right and keep 1st element as identity 0
            int k = threadIdx.x + blockIdx.x * blockDim.x;
            if (k >= n) {
                return;
            }
            if (k == 0) {
                odata[k] = 0;
            }
            else {
                odata[k] = idata[k - 1];
            }
        }
        
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startGpuTimer();

            //// DEBUGGER TEST
            // int noOfBlocks = 1;
            // dim3 blockSize(32, 32);
            // kernTestDebugger << < noOfBlocks, blockSize >> > (2);
            // 
            
            // TODO
            int* dev_buffer1;
            int* dev_buffer2;

            dim3 gridSize(32, 32);
            dim3 blockSize(32, 32);

            // Memory allocation
            cudaMalloc((void**)&dev_buffer1, sizeof(int) * n);
            checkCUDAError("cudaMalloc dev_idata failed!");
            cudaMalloc((void**)&dev_buffer2, sizeof(int) * n);
            checkCUDAError("cudaMalloc dev_odata failed!");
            cudaMemcpy(dev_buffer1, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
            
            int maxDepth = ilog2ceil(n);
            for (int d = 1; d <= maxDepth; d++) {    // where d is depth of iteration
                kernNaiveScan << <gridSize, blockSize >> > (n, d, dev_buffer2, dev_buffer1);
                cudaMemcpy(dev_buffer1, dev_buffer2, sizeof(int) * n, cudaMemcpyDeviceToDevice);
            }

            // converting from inclusive to exclusive scan using same buffers
            kernInclusiveToExclusive << <gridSize, blockSize >> > (n, dev_buffer1, dev_buffer2);
            cudaMemcpy(odata, dev_buffer1, sizeof(int) * (n), cudaMemcpyDeviceToHost);
            
            cudaFree(dev_buffer1);
            cudaFree(dev_buffer2);

            timer().endGpuTimer();

        }
    }
}
