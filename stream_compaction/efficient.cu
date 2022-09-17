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
        void upDownSweep(int n, int* data, dim3 blockPerGrid)
        {
            for (int d = 0; d <= ilog2ceil(n) - 1; ++d) {
                kernUpSweep << < blockPerGrid, blockSize >> > (n, d, data);
            }
            cudaDeviceSynchronize();

            cudaMemset(data + n - 1, 0, sizeof(int));
            checkCUDAError("cudaMemset failed!");


            for (int d = ilog2ceil(n) - 1; d >= 0; --d) {
                kernDownSweep << < blockPerGrid, blockSize >> > (n, d, data);
            }
            cudaDeviceSynchronize();
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

            upDownSweep(intermArraySize, dev_data, fullBlocksPerGrid);
            
            timer().endGpuTimer();
            cudaMemcpy(odata, dev_data, sizeof(int) * n, cudaMemcpyDeviceToHost);
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
        int compact(int n, int* odata, const int* idata) {
            dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);
            int arraySize = 1 << ilog2ceil(n);

            int* dev_indices;
            int* dev_bool;
            int* dev_idata;
            int* dev_odata;
            cudaMalloc((void**)&dev_indices, arraySize * sizeof(int));
            checkCUDAError("cudaMalloc dev_indices failed!");
            cudaMalloc((void**)&dev_bool, arraySize * sizeof(int));
            checkCUDAError("cudaMalloc dev_bool failed!");
            cudaMalloc((void**)&dev_idata, arraySize * sizeof(int));
            checkCUDAError("cudaMalloc dev_idata failed!");

            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy dev_bool to dev_data failed!");


            timer().startGpuTimer();
            // Step 1
            StreamCompaction::Common::kernMapToBoolean << <fullBlocksPerGrid, blockSize >> > (n, dev_bool, dev_idata);
            cudaDeviceSynchronize();
            
            cudaMemcpy(dev_indices, dev_bool, n * sizeof(int), cudaMemcpyDeviceToDevice);
            checkCUDAError("cudaMemcpy dev_bool to dev_data failed!");


            // Step 2
            upDownSweep(arraySize, dev_indices, fullBlocksPerGrid);

            int returnSize = 0;
            cudaMemcpy(&returnSize, dev_indices + arraySize - 1, sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy dev_indices to host failed!");

            // Di shared this edge case with me
            // When the input array has the last element non-zero, it will fail
            // hence we can add the last bit of the bool array to the return size to make sure that this case is covered
            int lastBool = 0;
            cudaMemcpy(&lastBool, dev_bool + arraySize - 1, sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy dev_bool to host failed!");
            returnSize += lastBool;


            cudaMalloc((void**)&dev_odata, returnSize * sizeof(int));
            checkCUDAError("cudaMalloc dev_odata failed!");


            // Step 3
             StreamCompaction::Common::kernScatter <<<fullBlocksPerGrid, blockSize >>>(arraySize, dev_odata,
                 dev_idata, dev_bool, dev_indices);
             cudaDeviceSynchronize();


            timer().endGpuTimer();

            cudaMemcpy(odata, dev_odata, returnSize * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy dev_odata to host failed!");
            
            cudaFree(dev_indices);
            cudaFree(dev_odata);
            cudaFree(dev_idata);
            cudaFree(dev_bool);

            return returnSize;
        }
    }
}
