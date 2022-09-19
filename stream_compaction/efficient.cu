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

        __global__ void kernUpSweepOpt(int n, int depth, int offset, int* data)
        {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            int offset_1 = offset << 1;
            int new_index = index * offset_1 + offset_1 - 1;
            if (index > (n-1))
            {
                return;
            }
            data[new_index] += data[index * offset_1 + offset - 1];

        }
        __global__ void kernDownSweepOpt(int n, int depth, int offset, int* data)
        {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            int offset_1 = offset << 1;
            int parent_i = index * offset_1 + offset_1 - 1;
            int left_i = index * offset_1 + offset - 1;
            if (index > (n-1))
            {
                return;
            }
            if (n == 1)
            {
                data[parent_i] = 0;
            }
            int t = data[left_i];
            data[left_i] = data[parent_i];
            data[parent_i] += t;
          

        }

        //__global__ void kernUpSweep(int n, int depth, int offset, int* data)
        //{
        //    int index = threadIdx.x + (blockIdx.x * blockDim.x);
        //    if (index > n)
        //    {
        //        return;
        //    }
        //    if (((index + 1) % (1 << (depth + 1))) == 0)
        //    {
        //        data[index] += data[index - offset];
        //    }

        //}
        //__global__ void kernDownSweep(int n, int depth, int offset, int* data, bool root)
        //{
        //    int index = threadIdx.x + (blockIdx.x * blockDim.x);
        //    if (index > n)
        //    {
        //        return;
        //    }
        //    if (index == (n-1) && root)
        //    {
        //        data[index] = 0;
        //    }
        //    if (((index + 1) % (1 << (depth + 1))) == 0)
        //    {
        //        int t = data[index - offset];
        //        data[index - offset] = data[index];
        //        data[index] += t;
        //    }

        //}

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        //void scanNoOpt(int n, int *odata, const int *idata) {
        //    int blockSize = 128;
        //    int numBlocks = ((n + blockSize - 1) / blockSize);
        //    int* dev_data;
        //    float d = ilog2ceil(n);
        //    int pot = pow(2, d);
        //    cudaMalloc((void**)&dev_data, pot * sizeof(int));
        //    cudaMemset(dev_data, 0, pot * sizeof(int));
        //    cudaMemcpy(dev_data, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
        //    cudaDeviceSynchronize();
        //    timer().startGpuTimer();
        //    for (int depth = 0; depth < d; depth++)
        //    {
        //        int offset = 1 << depth;
        //        kernUpSweep << < numBlocks, blockSize >> > (pot, depth, offset, dev_data);
        //        
        //    }
        //    bool root = true;
        //    cudaDeviceSynchronize();
        //    for (int depth = d - 1; depth >= 0; depth--)
        //    {
        //        int offset = 1 << depth;
        //        kernDownSweep << < numBlocks, blockSize >> > (pot, depth, offset, dev_data, root);
        //        root = false;
        //    }
        //    timer().endGpuTimer();
        //    cudaMemcpy(odata, dev_data, sizeof(int) * n, cudaMemcpyDeviceToHost);
        //    cudaFree(dev_data);
        //}

        void scan(int n, int* odata, const int* idata) {
            int blockSize = 64;
            int* dev_data;
            int d = ilog2ceil(n);
            int pot = 1 << d;
            int num = pot;
            cudaMalloc((void**)&dev_data, pot * sizeof(int));
            checkCUDAError("Malloc dev_data Failed! ");
            cudaMemset(dev_data, 0, pot * sizeof(int));
            checkCUDAError("Memset dev_data Failed! ");
            cudaMemcpy(dev_data, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
            checkCUDAError("Memcpy dev_data Failed! ");
            cudaDeviceSynchronize();
            
            timer().startGpuTimer();
            
            for (int depth = 0; depth < d; depth++)
            {
                num /= 2;
                int offset = 1 << depth;
                int numBlocks = ((num + blockSize - 1) / blockSize);
                kernUpSweepOpt << < numBlocks, blockSize >> > (num, depth, offset, dev_data);

            }
            cudaDeviceSynchronize();
            for (int depth = d - 1; depth >= 0; depth--)
            {
                int offset = 1 << depth;
                int numBlocks = ((num + blockSize - 1) / blockSize);
                kernDownSweepOpt << < numBlocks, blockSize >> > (num, depth, offset, dev_data);
                num *= 2;
            }
            timer().endGpuTimer();
            cudaMemcpy(odata, dev_data, sizeof(int) * n, cudaMemcpyDeviceToHost);
            checkCUDAError("Memcpy dev_data back Failed! ");
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
        int compact(int n, int *odata, const int *idata) {
            int blockSize = 64;
            int numBlocks = ((n + blockSize - 1) / blockSize);
            int* dev_idata;
            int* dev_odata;
            int* dev_bools;
            int* dev_indicies;
            int d = ilog2ceil(n);
            int pot = 1 << d;
            int num = pot;
            cudaMalloc((void**)&dev_idata, pot * sizeof(int));
            checkCUDAError("Malloc dev_idata Failed! ");
            cudaMalloc((void**)&dev_odata, pot * sizeof(int));
            checkCUDAError("Malloc dev_odata Failed! ");
            cudaMalloc((void**)&dev_bools, pot * sizeof(int));
            checkCUDAError("Malloc dev_bools Failed! ");
            cudaMalloc((void**)&dev_indicies, pot * sizeof(int));
            checkCUDAError("Malloc dev_indices Failed! ");
            cudaMemset(dev_idata, 0, pot * sizeof(int));
            checkCUDAError("Memset dev_idata Failed! ");
            cudaMemset(dev_odata, 0, pot * sizeof(int));
            checkCUDAError("Memset dev_odata Failed! ");
            cudaMemcpy(dev_idata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
            checkCUDAError("Memcpy dev_idata Failed! ");
            cudaDeviceSynchronize();
            timer().startGpuTimer();

            Common::kernMapToBoolean << < numBlocks, blockSize >> > (n, dev_bools, dev_idata);
            cudaMemcpy(dev_indicies, dev_bools, sizeof(int) * n, cudaMemcpyDeviceToDevice);
            for (int depth = 0; depth < d; depth++)
            {
                num /= 2;
                int offset = 1 << depth;
                int numBlocks = ((num + blockSize - 1) / blockSize);
                kernUpSweepOpt << < numBlocks, blockSize >> > (num, depth, offset, dev_indicies);

            }
            cudaDeviceSynchronize();
            for (int depth = d - 1; depth >= 0; depth--)
            {
                int offset = 1 << depth;
                int numBlocks = ((num + blockSize - 1) / blockSize);
                kernDownSweepOpt << < numBlocks, blockSize >> > (num, depth, offset, dev_indicies);
                num *= 2;
            }
            Common::kernScatter << < numBlocks, blockSize >> > (n, dev_odata, dev_idata, dev_bools, dev_indicies);
            timer().endGpuTimer();
            cudaMemcpy(odata, dev_odata, sizeof(int) * n, cudaMemcpyDeviceToHost);
            checkCUDAError("Memcpy dev_odata Failed! ");
            int num_elements = 0;
            for (int i = 0; i < n; i++)
            {
                if (odata[i] != 0)
                {
                    num_elements++;
                }
            }
            cudaFree(dev_idata);
            cudaFree(dev_odata);
            cudaFree(dev_bools);
            cudaFree(dev_indicies);
            return num_elements;
        }
    }
}
