#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

#include <iostream> // PLEASE REMOVE THIS AFTER TESTING

/*! Block size used for CUDA kernel launch. */
#define blockSize 512

#define OPTIMIZED

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
#ifdef OPTIMIZED
            if (index < stride) {
                odata[offset_d * (2 * index + 2) - 1] += odata[offset_d * (2 * index + 1) - 1];
            }
#else
            if ((index % stride) == 0) {
                odata[index + offset_d_one - 1] += odata[index + offset_d - 1];
            }
#endif
        }

        __global__ void kernDownsweep(int n, int d, int stride, int offset_d_one, int offset_d, int* odata) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
#ifdef OPTIMIZED
            if (index < d) {
                // Old left child
                int t = odata[stride * (2 * index + 1) - 1];
                // Set left child to current node value
                odata[stride * (2 * index + 1) - 1] = odata[stride * (2 * index + 2) - 1];
                // Set right child = old left child + current node value
                odata[stride * (2 * index + 2) - 1] += t;
            }
#else
            if ((index % stride) == 0) {
                // Old left child
                int t = odata[index + offset_d - 1];
                // Set left child to current node value
                odata[index + offset_d - 1] = odata[index + offset_d_one - 1];
                // Set right child = old left child + current node value
                odata[index + offset_d_one - 1] += t;
            }
#endif
        }
        /**
         * Helper function for "scan" to avoid conflicts with global timers.
         */
        void prefixSum(int n, int *odata, const int *idata) {
            int log2_n = ilog2ceil(n);
            int N = pow(2, log2_n); // next highest power of 2
            dim3 blocksPerGrid((N + blockSize - 1) / blockSize);

            int* dev_scan_output;

            // Allocate device memory (size 2^log2n)
            cudaMalloc((void**)&dev_scan_output, N * sizeof(int));
            checkCUDAErrorFn("cudaMalloc dev_scan_output failed!");

            cudaMemset(dev_scan_output, 0, N * sizeof(int));
            checkCUDAErrorFn("second cudaMemset dev_scan_output failed!");

            // Copy data to the GPU
            cudaMemcpy(dev_scan_output, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAErrorFn("memcpy to GPU failed!");

            timer().startGpuTimer();
            // Perform upsweep (inclusive scan)  
#ifdef OPTIMIZED
            int stride = N / 2;
#else
            int stride = 2;
#endif
            for (int d = 0; d < log2_n; ++d) {
                int offset_d_one = pow(2, d + 1);
                int offset_d = pow(2, d);
                kernUpsweep << <blocksPerGrid, blockSize >> > (N, stride, offset_d_one, offset_d, dev_scan_output);
#ifdef OPTIMIZED
                stride /= 2;
#else
                stride *= 2;
#endif
            }
            
            // Set "root" (last element in array) to 0
            cudaMemset(dev_scan_output + N - 1, 0, sizeof(int));
            checkCUDAErrorFn("first cudaMemset dev_scan_output failed!");

            // Perform downsweep
#ifdef OPTIMIZED
            stride = N / 2;
            for (int d = 1; d < N; d *= 2) {
#else
            stride /= 2;
            for (int d = log2_n - 1; d >= 0; --d) {
#endif
                int offset_d_one = pow(2, d + 1);
                int offset_d = pow(2, d);
                kernDownsweep << <blocksPerGrid, blockSize >> > (N, d, stride, offset_d_one, offset_d, dev_scan_output);
                stride /= 2;
            }
            timer().endGpuTimer();

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
            prefixSum(n, odata, idata);
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
            int log2_n = ilog2ceil(n);
            int N = pow(2, log2_n); // next highest power of 2
            dim3 blocksPerGrid((N + blockSize - 1) / blockSize);

            int elements[1];
            int* dev_bool_input;
            int* dev_bools;
            int* dev_scatter_input;
            int* dev_scatter_output;

            // Allocate device memory (size 2^log2n)
            cudaMalloc((void**)&dev_bool_input, N * sizeof(int));
            checkCUDAErrorFn("cudaMalloc dev_scan_temp failed!");
            cudaMalloc((void**)&dev_bools, N * sizeof(int));
            checkCUDAErrorFn("cudaMalloc dev_scan_temp failed!");
            cudaMalloc((void**)&dev_scatter_input, N * sizeof(int));
            checkCUDAErrorFn("cudaMalloc dev_scatter_input failed!");
            cudaMalloc((void**)&dev_scatter_output, N * sizeof(int));
            checkCUDAErrorFn("cudaMalloc dev_scatter_output failed!");

            // Copy data to the GPU
            cudaMemcpy(dev_bool_input, idata, N * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAErrorFn("memcpy to GPU failed!");

            // Zero the scan output array (serves as extra padding for non-power-of-two arrays)
            cudaMemset(dev_bools, 0, N * sizeof(int));
            cudaMemset(dev_scatter_input, 0, N * sizeof(int));
            checkCUDAErrorFn("cudaMemset dev_scatter_input failed!");
            
            // Make temporary array with 0s / 1s to indicate if data meets criteria
            timer().startGpuTimer();
            StreamCompaction::Common::kernMapToBoolean<<<blocksPerGrid, blockSize >> > (n, dev_bools, dev_bool_input);
            cudaMemcpy(dev_scatter_input, dev_bools, n * sizeof(int), cudaMemcpyDeviceToDevice);
            checkCUDAErrorFn("memcpy to GPU failed!");

            // Run exclusive scan on temporary array
            // Upsweep   
#ifdef OPTIMIZED
            int stride = N / 2;
#else
            int stride = 2;
#endif
            for (int d = 0; d < log2_n; ++d) {
                int offset_d_one = pow(2, d + 1);
                int offset_d = pow(2, d);
                kernUpsweep << <blocksPerGrid, blockSize >> > (N, stride, offset_d_one, offset_d, dev_scatter_input);
#ifdef OPTIMIZED
                stride /= 2;
#else
                stride *= 2;
#endif
            }

            // Get element count
            cudaMemcpy(&elements, dev_scatter_input + N - 1, sizeof(int), cudaMemcpyDeviceToHost);

            // Set "root" (last element in array) to 0
            cudaMemset(dev_scatter_input + N - 1, 0, sizeof(int));
            checkCUDAErrorFn("cudaMemset dev_scatter_input failed!");

            // Downsweep
#ifdef OPTIMIZED
            stride = N / 2;
            for (int d = 1; d < N; d *= 2) {
#else
            stride /= 2;
            for (int d = log2_n - 1; d >= 0; --d) {
#endif
                int offset_d_one = pow(2, d + 1);
                int offset_d = pow(2, d);
                kernDownsweep << <blocksPerGrid, blockSize >> > (N, d, stride, offset_d_one, offset_d, dev_scatter_input);
                stride /= 2;
            }

            // Scatter
            StreamCompaction::Common::kernScatter << <blocksPerGrid, blockSize >> > (N, dev_scatter_output, dev_bool_input, dev_bools, dev_scatter_input);
            timer().endGpuTimer();

            // Copy data back to CPU
            cudaMemcpy(odata, dev_scatter_output, N * sizeof(int), cudaMemcpyDeviceToHost);
                        
            // Cleanup memory
            cudaFree(dev_bool_input);
            cudaFree(dev_bools);
            cudaFree(dev_scatter_input);
            cudaFree(dev_scatter_output);
            checkCUDAErrorFn("cudaFree failed!");

            return elements[0];
        }
    }
}
