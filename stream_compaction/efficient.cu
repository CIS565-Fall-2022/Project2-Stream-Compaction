#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

#include <iostream> // PLEASE REMOVE THIS AFTER TESTING

/*! Block size used for CUDA kernel launch. */
#define blockSize 128

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernUpsweep(int n, int stride, int offset_d_one, int offset_d, int* odata) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n) {
                return;
            }

            if ((index % stride) == 0) {
                odata[index + offset_d_one - 1] += odata[index + offset_d - 1];
            }
        }

        //__global__ void kernDownsweep(int n, int* odata, const int* idata) {
        //    int index = threadIdx.x + (blockIdx.x * blockDim.x);
        //    if (index >= n) {
        //        return;
        //    }

        //    if (index >= offset) {
        //        temp[index] += temp[index - offset];
        //    }
        //    else {
        //        temp[index] = temp[index];
        //    }
        //    odata[index] = temp[index];
        //}

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
            dim3 blocksPerGrid((n + blockSize - 1) / blockSize);

            int* dev_scan_input;
            int* dev_scan_output;

            // Allocate device memory
            cudaMalloc((void**)&dev_scan_output, n * sizeof(int));
            checkCUDAErrorFn("cudaMalloc dev_scan_output failed!");

            // Copy data to the GPU
            cudaMemcpy(dev_scan_output, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
            checkCUDAErrorFn("memcpy to GPU failed!");

            // Perform upsweep (inclusive scan)
            int log2_n = ilog2ceil(n);
            int stride = 2;
            for (int d = 0; d < log2_n; ++d) {
                int offset_d_one = pow(2, d + 1);
                int offset_d = pow(2, d);
                kernUpsweep << <blocksPerGrid, blockSize >> > (n, stride, offset_d_one, offset_d, dev_scan_output);
                stride *= 2;
            }

            // Perform downsweep

            // Copy data back to the CPU
            cudaMemcpy(odata, dev_scan_output, sizeof(int) * n, cudaMemcpyDeviceToHost);
            checkCUDAErrorFn("memcpy to CPU failed!");

            // Cleanup memory
            cudaFree(dev_scan_output);
            checkCUDAErrorFn("cudaFree failed!");

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
