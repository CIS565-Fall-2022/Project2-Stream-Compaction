#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kern_ComputeExclusiveScanIteration(int N, int offset, int* odata, int* idata) {
            int thread_num = threadIdx.x + (blockIdx.x * blockDim.x);
            if (thread_num >= N) {
                return;
            }

            // warp 0 will have divergent branches on offest == 1 but that would happen if moved to seperate kernel anyways
            if (offset == 1 && thread_num == 0) {
                // SPECIAL CASE: first iteration and 0th element should be zeroed-out for exclusive scan
                odata[thread_num] = 0;
            }
            else if (offset == 1 && thread_num == 1) {
                // SPECIAL CASE: first iteration and 1st element should be equal to the original 0th element for exclusive scan
                odata[thread_num] = idata[thread_num - 1];
            }
            else if (offset == 1 && thread_num > 1) {
                // SPECIAL CASE: first iteration elements past index 1 should use two previous values for summing
                odata[thread_num] = idata[thread_num - 2] + idata[thread_num - 1];
            }
            else if (thread_num < offset) {
                // same as for inclusive
                odata[thread_num] = idata[thread_num];
            }
            else {
                // same as for inclusive
                odata[thread_num] = idata[thread_num - offset] + idata[thread_num];
            }
        }

        __global__ void kern_ComputeInclusiveScanIteration(int N, int offset, int* odata, int* idata) {
            int thread_num = threadIdx.x + (blockIdx.x * blockDim.x);
            if (thread_num >= N) {
                return;
            }
            

            if (thread_num < offset) {
                odata[thread_num] = idata[thread_num];
            }
            else {
                odata[thread_num] = idata[thread_num - offset] + idata[thread_num];
            }
        }

        __global__ void kern_InclusiveToExclusiveScan(int N, int* odata, int* idata) {
            int thread_num = threadIdx.x + (blockIdx.x * blockDim.x);
            if (thread_num >= N) {
                return;
            }

            if (thread_num == 0) {
                odata[thread_num] = 0;
            }
            else {
                odata[thread_num] = idata[thread_num - 1];
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            int* dev_odata_1;
            int* dev_odata_2;

            cudaMalloc((void**)&dev_odata_1, n * sizeof(int));
            cudaMalloc((void**)&dev_odata_2, n * sizeof(int));
            cudaDeviceSynchronize();

            cudaMemcpy(dev_odata_1, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
            cudaDeviceSynchronize();

            timer().startGpuTimer();

            int numBlocks = (n + blockSize - 1) / blockSize;

            int num_iterations = ilog2ceil(n);
            for (int d = 1; d <= num_iterations; ++d) {
                // compute scan iteration d
#ifdef inclusiveToExclusive
                kern_ComputeInclusiveScanIteration <<< numBlocks, blockSize >>> (n, pow(2, d - 1), dev_odata_2, dev_odata_1);
#else
                kern_ComputeExclusiveScanIteration <<< numBlocks, blockSize >>> (n, pow(2, d - 1), dev_odata_2, dev_odata_1);
#endif
                cudaDeviceSynchronize();
                int* temp = dev_odata_1;
                dev_odata_1 = dev_odata_2;
                dev_odata_2 = temp;
            }

#ifdef inclusiveToExclusive
            kern_InclusiveToExclusiveScan <<< numBlocks, blockSize >>> (n, dev_odata_2, dev_odata_1);
            cudaDeviceSynchronize();
#endif
            timer().endGpuTimer();

            cudaMemcpy(odata, dev_odata_1, sizeof(int) * n, cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();

            cudaFree(dev_odata_1);
            cudaFree(dev_odata_2);
            cudaDeviceSynchronize();
        }
    }
}
