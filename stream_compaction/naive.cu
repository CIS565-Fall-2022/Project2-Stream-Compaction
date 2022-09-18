#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

#include <iostream> // PLEASE REMOVE THIS AFTER TESTING

/*! Block size used for CUDA kernel launch. */
#define blockSize 128

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        /**
         * Shifts the array one to the right.
         */
        __global__ void kernInclToExcl(int n, int* odata, const int* idata) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n) {
                return;
            }

            odata[index] = (index > 0) ? idata[index - 1] : 0;
        }
        
        /**
         * Performs one thread's work for a naive parallel prefix-sum.
         */
        __global__ void kernNaiveScan(int n, int d, int offset, int* odata, const int* idata) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n) {
                return;
            }

            if (index >= offset) {
                odata[index] = idata[index - offset] + idata[index];
            }
            else {
                odata[index] = idata[index];
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            // Determine block size
            dim3 blocksPerGrid((n + blockSize - 1) / blockSize);

            // Allocate seperate arrays to hold results between iterations
            int* dev_scan_input;
            int* dev_scan_output;

            // Allocate device memory
            cudaMalloc((void**)&dev_scan_input, n * sizeof(int));
            checkCUDAErrorFn("cudaMalloc dev_scan_input failed!");
            cudaMalloc((void**)&dev_scan_output, n * sizeof(int));
            checkCUDAErrorFn("cudaMalloc dev_scan_output failed!");

            // Copy data to the GPU
            cudaMemcpy(dev_scan_input, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
            cudaMemcpy(dev_scan_output, odata, sizeof(int) * n, cudaMemcpyHostToDevice);
            checkCUDAErrorFn("memcpy to GPU failed!");

            timer().startGpuTimer();
            // Transform inclusive array to exclusive array
            kernInclToExcl << <blocksPerGrid, blockSize >> > (n, dev_scan_output, dev_scan_input);

            // Perform exclusive scan
            int log2_n = ilog2ceil(n);
            int offset = 1;
            for (int d = 1; d <= log2_n; ++d) {
                kernNaiveScan << <blocksPerGrid, blockSize >> > (n, d, offset, dev_scan_input, dev_scan_output);
                std::swap(dev_scan_input, dev_scan_output);
                offset *= 2;
            }
            timer().endGpuTimer();

            // Copy data back to the CPU
            cudaMemcpy(odata, dev_scan_output, sizeof(int) * n, cudaMemcpyDeviceToHost);
            checkCUDAErrorFn("memcpy to CPU failed!");

            //// Print output vals
            //std::cout << "After scan: " << std::endl;
            //for (int i = 0; i < n; i++) {
            //    std::cout << "  scan[" << i << "]: " << odata[i] << std::endl;
            //}

            // Cleanup memory
            cudaFree(dev_scan_input);
            cudaFree(dev_scan_output);
            checkCUDAErrorFn("cudaFree failed!");
        }
    }
}
