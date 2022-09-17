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

        __global__ void kernDownsweep(int n, int stride, int offset_d_one, int offset_d, int* odata) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n) {
                return;
            }

            if ((index % stride) == 0) {
                // Old left child
                int t = odata[index + offset_d - 1];
                // Set left child to current node value
                odata[index + offset_d - 1] = odata[index + offset_d_one - 1];
                // Set right child = old left child + current node value
                odata[index + offset_d_one - 1] += t;
            }
        }

        __global__ void kernMeetsCriteria(int n, int* temp, const int* idata) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n) {
                return;
            }

            temp[index] = (idata[index] != 0) ? 1 : 0;
        }

        __global__ void kernScatter(int n, int* odata, const int* idx, const int* temp, const int* idata) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n) {
                return;
            }

            int meets_criteria = temp[index];
            int final_idx = idx[index];
            if (meets_criteria) {
                odata[final_idx] = idata[index];
            }
        }

        /**
         * Helper function for "scan" to avoid conflicts with global timers.
         */
        void prefixSum(int n, int &elements, int *odata, const int *idata) {
            dim3 blocksPerGrid((n + blockSize - 1) / blockSize);
            int log2_n = ilog2ceil(n);
            int N = pow(2, log2_n); // next highest power of 2
            int array_offset = N - n;

            int* dev_scan_output;

            // Allocate device memory (size 2^log2n)
            cudaMalloc((void**)&dev_scan_output, N * sizeof(int));
            checkCUDAErrorFn("cudaMalloc dev_scan_output failed!");

            cudaMemset(dev_scan_output, 0, N * sizeof(int));
            checkCUDAErrorFn("cudaMemset dev_scan_output failed!");

            // Copy data to the GPU
            cudaMemcpy(dev_scan_output, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAErrorFn("memcpy to GPU failed!");

            // Perform upsweep (inclusive scan)   
            int stride = 2;
            for (int d = 0; d < log2_n; ++d) {
                int offset_d_one = pow(2, d + 1);
                int offset_d = pow(2, d);
                kernUpsweep << <blocksPerGrid, blockSize >> > (N, stride, offset_d_one, offset_d, dev_scan_output);
                stride *= 2;
            }

            // Copy data back to the CPU
            cudaMemcpy(odata, dev_scan_output, N * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAErrorFn("memcpy to CPU failed!");
            
            // Perform downsweep
            // Set "root" (last element in array) to 0
            elements = odata[N - 1];
            odata[N - 1] = 0;
            cudaMemcpy(dev_scan_output, odata, N * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAErrorFn("memcpy to GPU failed!");

            stride /= 2;
            for (int d = log2_n - 1; d >= 0; --d) {
                int offset_d_one = pow(2, d + 1);
                int offset_d = pow(2, d);
                kernDownsweep << <blocksPerGrid, blockSize >> > (N, stride, offset_d_one, offset_d, dev_scan_output);
                stride /= 2;
            }

            cudaMemcpy(odata, dev_scan_output, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAErrorFn("memcpy to CPU failed!");

            // Cleanup memory
            cudaFree(dev_scan_output);
            checkCUDAErrorFn("cudaFree failed!");
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
            int elements = -1;
            prefixSum(n, elements, odata, idata);
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
            dim3 blocksPerGrid((n + blockSize - 1) / blockSize);
            int log2_n = ilog2ceil(n);
            int N = pow(2, log2_n); // next highest power of 2
            int elements = -1;

            int* host_scan_input = new int[N];
            int* host_scan_output = new int[N];
            int* dev_scan_input;
            int* dev_scan_temp;
            int* dev_scatter_input;
            int* dev_scatter_output;

            // Allocate device memory (size 2^log2n)
            cudaMalloc((void**)&dev_scan_input, N * sizeof(int));
            checkCUDAErrorFn("cudaMalloc dev_scan_temp failed!");
            cudaMalloc((void**)&dev_scan_temp, N * sizeof(int));
            checkCUDAErrorFn("cudaMalloc dev_scan_temp failed!");
            cudaMalloc((void**)&dev_scatter_input, N * sizeof(int));
            checkCUDAErrorFn("cudaMalloc dev_scatter_input failed!");
            cudaMalloc((void**)&dev_scatter_output, N * sizeof(int));
            checkCUDAErrorFn("cudaMalloc dev_scatter_output failed!");

            // Copy data to the GPU
            cudaMemcpy(dev_scan_input, idata, N * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAErrorFn("memcpy to GPU failed!");

            // Make temporary array with 0s / 1s to indicate if data meets criteria
            kernMeetsCriteria<<<blocksPerGrid, blockSize >> > (N, dev_scan_temp, dev_scan_input);

            // Run exclusive scan on temporary array
            cudaMemcpy(host_scan_input, dev_scan_temp, N * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAErrorFn("memcpy to CPU failed!");
            prefixSum(n, elements, host_scan_output, host_scan_input);

            // Scatter
            cudaMemcpy(dev_scatter_input, host_scan_output, N * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(dev_scatter_output, odata, N * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAErrorFn("memcpy to GPU failed!");

            kernScatter << <blocksPerGrid, blockSize >> > (n, dev_scatter_output, dev_scatter_input, dev_scan_temp, dev_scan_input);
            cudaMemcpy(odata, dev_scatter_output, N * sizeof(int), cudaMemcpyDeviceToHost);
                        
            // Cleanup memory
            cudaFree(dev_scan_input);
            cudaFree(dev_scan_temp);
            cudaFree(dev_scatter_input);
            cudaFree(dev_scatter_output);
            checkCUDAErrorFn("cudaFree failed!");

            delete host_scan_input, host_scan_output;

            timer().endGpuTimer();
            return elements;
        }
    }
}
