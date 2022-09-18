#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

#include <bitset>
#include "device_launch_parameters.h"

#define NAIVE 1
#define NUM_BANKS 16 
#define LOG_NUM_BANKS 4 
#define CONFLICT_FREE_OFFSET(n) ((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS)) 

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernUpSweep(int n, int stride, int* data)
        {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index >= n) return;
            if (index % stride != 0 || index + stride -1  >= n) return;
            data[index + stride - 1] += data[index + stride / 2 - 1];
        }

        __global__ void kernDownSweep(int n, int stride, int* data)
        {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index >= n) return;
            if (index % stride != 0 || index + stride -1 >= n) return;
            int temp = data[index + stride / 2 - 1];
            data[index + stride / 2 - 1] = data[index + stride - 1];
            data[index + stride - 1] += temp;
        }

        __global__ void kernOptimizedUpSweep(int n, int d, int offset, int* data)
        {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index >= n) return;
            if (index < d) {
                int ai = offset * (2 * index + 1) - 1;
                int bi = offset * (2 * index + 2) - 1;
                data[bi] += data[ai];
            }
        }

        __global__ void kernOptimizedDownSweep(int n, int d, int offset, int* data)
        {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index >= n) return;
            if (index < d) {
                int ai = offset * (2 * index + 1) - 1;
                int bi = offset * (2 * index + 2) - 1;
                int t = data[ai];
                data[ai] = data[bi];
                data[bi] += t;
            }
        }

        //adapted from GPU Gem 3
        __global__ void kernOptimizedPerBlockScan(int* g_odata, int* g_idata, int n) {
            // allocated on invocation 
            extern __shared__ int temp[];

            int thid = threadIdx.x;
            int blockOffset = blockIdx.x * blockDim.x;

            int index = blockOffset + thid;
            __syncthreads();
            temp[thid] = g_idata[index];

            //temp[2 * thid] = g_idata[2 * index];
            //temp[2 * thid + 1] = g_idata[2 * index + 1];

            // build sum in place up the tree 
            int offset = 1;
            for (int d = n >> 1; d > 0; d >>= 1)
            {
                __syncthreads();
                if (thid < d) {
                    //int ai = offset * (2 * thid + 1) - 1; 
                    //int bi = offset * (2 * thid + 2) - 1; 
                    //ai += CONFLICT_FREE_OFFSET(ai);
                    //bi += CONFLICT_FREE_OFFSET(bi);

                    int ai = offset * (2 * thid + 1) - 1;
                    int bi = offset * (2 * thid + 2) - 1;

                    temp[bi] += temp[ai];
                }
                offset *= 2;
            }
            // clear the last element
            if (thid == 0) { temp[n-1] = 0; }
            // traverse down tree & build scan 
            for (int d = 1; d < n; d *= 2)
            {
                offset >>= 1;
                __syncthreads();
                if (thid < d) {

                    int ai = offset * (2 * thid + 1) - 1;
                    int bi = offset * (2 * thid + 2) - 1;
                    int t = temp[ai];
                    temp[ai] = temp[bi];
                    temp[bi] += t;
                }
            }
            
            //g_odata[2 * thid + blockOffset] = temp[2 * thid];
            //g_odata[2 * thid + 1 + blockOffset] = temp[2 * thid + 1];

            //g_odata[2 * index] = temp[2 * thid];
            //g_odata[2 * index + 1] = temp[2 * thid + 1];
            __syncthreads();
            g_odata[index] = temp[thid];
        }

        __global__ void kernWriteSumArray(int* sum, int* odata)
        {
            int index = threadIdx.x;
            if (index == blockDim.x - 1)
                sum[blockIdx.x] = odata[(blockIdx.x + 1) * blockDim.x - 1];
        }

        __global__ void kernAddIncrement(int* outdata, int* sumArrayCopy)
        {
            int index = threadIdx.x;
            //for (int i = 0; i < blockDim.x; i++) {
            outdata[blockDim.x * blockIdx.x + index] += sumArrayCopy[blockIdx.x];
            //}
        }

        __global__ void kernMakeInclusive(int* inclusive, int* odata, int* idata)
        {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (threadIdx.x == blockDim.x - 1) {
                inclusive[index] = odata[index] + idata[index];
            }
            else {
                inclusive[index] = odata[index + 1];
            }
        }

        __global__ void kernMakeExclusive(int* odata, int* idata)
        {
            int index = blockDim.x * blockIdx.x + threadIdx.x;
            if (index == 0)
                odata[index] = 0;
            else
                odata[index] = idata[index - 1];
        }
        
        void scanInclusive(int n, int* odata, const int* idata)
        {
            //extend to pow of 2
            int Num = 1 << ilog2ceil(n);
            int blockSize = 128;
            int blockNum = (Num + blockSize - 1) / blockSize;

            int* dev_data;
            cudaMalloc((void**)&dev_data, Num * sizeof(int));
            cudaMemset(dev_data, 0, Num * sizeof(int));
            cudaMemcpy(dev_data, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            int* dev_inclusive;
            cudaMalloc((void**)&dev_inclusive, n * sizeof(int));

            int offset = 1;
            for (int d = Num >> 1; d > 0; d >>= 1) {
                blockNum = (d + blockSize - 1) / blockSize;
                kernOptimizedUpSweep << <blockNum, blockSize >> > (Num, d, offset, dev_data);
                offset <<= 1;
            }
            cudaDeviceSynchronize();
            //get the last sum for inclusive scan
            int lastSum = 0;
            cudaMemcpy(&lastSum, dev_data + Num - 1, sizeof(int), cudaMemcpyDeviceToHost);

            cudaMemset(dev_data + Num - 1, 0, sizeof(int));
            for (int d = 1; d < Num; d <<= 1) {
                blockNum = (d + blockSize - 1) / blockSize;
                offset >>= 1;
                kernOptimizedDownSweep << <blockNum, blockSize >> > (Num, d, offset, dev_data);
            }
            cudaMemcpy(odata, dev_inclusive, n * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(dev_data);
        }

        void scanWithoutTimer(int n, int* odata, const int* idata)
        {
            int Num = 1 << ilog2ceil(n);
            int blockSize = 128;
            int blockNum = (Num + blockSize - 1) / blockSize;

            int* dev_data;
            cudaMalloc((void**)&dev_data, Num * sizeof(int));
            cudaMemset(dev_data, 0, Num * sizeof(int));
            cudaMemcpy(dev_data, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            int offset = 1;
            for (int d = Num >> 1; d > 0; d >>= 1) {
                blockNum = (d + blockSize - 1) / blockSize;
                kernOptimizedUpSweep << <blockNum, blockSize >> > (Num, d, offset, dev_data);
                offset <<= 1;
            }
            cudaDeviceSynchronize();
            cudaMemset(dev_data + Num - 1, 0, sizeof(int));
            for (int d = 1; d < Num; d <<= 1) {
                blockNum = (d + blockSize - 1) / blockSize;
                offset >>= 1;
                kernOptimizedDownSweep << <blockNum, blockSize >> > (Num, d, offset, dev_data);
            }
            cudaMemcpy(odata, dev_data, n * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(dev_data);
        }

        void scanWithSharedMemory(int n, int* odata, const int* idata)
        {
            int Num = 1 << ilog2ceil(n);
            int blockSize = 64;
            if (Num <= 512)
                blockSize = Num/2;
            else
                blockSize = 256;
            int blockNum = Num  / blockSize;

            int* inData;
            int* outData;
            int* inclusiveData;
            cudaMalloc((void**)&inData, Num * sizeof(int));
            cudaMalloc((void**)&outData, Num * sizeof(int));
            cudaMalloc((void**)&inclusiveData, Num * sizeof(int));
            cudaMemset(inData, 0, Num * sizeof(int));
            cudaMemcpy(inData, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            int* sumArray;
            cudaMalloc((void**)&sumArray, blockNum * sizeof(int));
            int* sumArrayCopy;
            cudaMalloc((void**)&sumArrayCopy, blockNum * sizeof(int));


            timer().startGpuTimer();
            //run scan on each block
            kernOptimizedPerBlockScan<<<blockNum, blockSize, blockSize * sizeof(int)>>> (outData, inData, blockSize);
            cudaDeviceSynchronize();

            //cudaMemcpy(odata, outData, n * sizeof(int), cudaMemcpyDeviceToHost);

            kernMakeInclusive << <blockNum, blockSize >> > (inclusiveData, outData, inData);

            //write total sum of each block into a new array
            kernWriteSumArray << <blockNum, blockSize >> > (sumArray, inclusiveData);
            cudaDeviceSynchronize();

            //exclusive scan
            scanWithoutTimer(blockNum, sumArrayCopy, sumArray);
            cudaDeviceSynchronize();

            //add block increment
            kernAddIncrement << <blockNum, blockSize >> > (inclusiveData, sumArrayCopy);
            cudaDeviceSynchronize();

            timer().endGpuTimer();

            kernMakeExclusive << <blockNum, blockSize >> > (inData, inclusiveData);
            
            cudaMemcpy(odata, inData, n * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(inData);
            cudaFree(outData);
            cudaFree(inclusiveData);
            cudaFree(sumArray);
            cudaFree(sumArrayCopy);
        }

        
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            //extend to pow of 2
            int Num = 1 << ilog2ceil(n);
            int blockSize = 128;
            int blockNum = (Num + blockSize - 1) / blockSize;
            
            int* dev_data;
            cudaMalloc((void**)&dev_data, Num * sizeof(int));
            cudaMemset(dev_data, 0, Num * sizeof(int));
            cudaMemcpy(dev_data, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            int offset = 1;
            timer().startGpuTimer();
            // TODO
            ////up-sweep
            //for (int i = 0; i <= ilog2ceil(Num); i++) {
            //    kernUpSweep << <blockNum, blockSize >> > (Num, pow(2, i + 1), dev_data);
            //}
            // cudaDeviceSynchronize()
            // cudaMemset(dev_data + Num - 1, 0, sizeof(int));
            ////down-sweep
            //for (int i = ilog2ceil(Num); i >= 0; i--) {
            //    kernDownSweep << <blockNum, blockSize >> > (Num, pow(2, i + 1), dev_data);
            //}
            for (int d = Num >> 1; d > 0; d >>= 1) {
                blockNum = (d + blockSize - 1) / blockSize;
                kernOptimizedUpSweep << <blockNum, blockSize >> > (Num,d, offset, dev_data);
                offset <<= 1;
            }
            cudaDeviceSynchronize();
            cudaMemset(dev_data + Num - 1, 0, sizeof(int));
            for (int d = 1; d < Num; d <<= 1) {
                blockNum = (d + blockSize - 1) / blockSize;
                offset >>= 1;
                kernOptimizedDownSweep << <blockNum, blockSize >> > (Num, d, offset, dev_data);
            }
            timer().endGpuTimer();
            cudaMemcpy(odata, dev_data, n * sizeof(int), cudaMemcpyDeviceToHost);
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
            int Num = pow(2, ilog2ceil(n));

            int* dev_bools;
            int* dev_scan_Result;
            int* dev_dataCopy;
            int* dev_ScatterResult;
            cudaMalloc((void**)&dev_dataCopy, Num * sizeof(int));
            cudaMalloc((void**)&dev_bools, Num * sizeof(int));
            cudaMalloc((void**)&dev_scan_Result, Num * sizeof(int));
            cudaMalloc((void**)&dev_ScatterResult, Num * sizeof(int));
            
            cudaMemset(dev_dataCopy, 0, Num * sizeof(int));
            cudaMemcpy(dev_dataCopy, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            dim3 blockSize(128);
            dim3 blockNum((n + blockSize.x - 1) / blockSize.x);


            timer().startGpuTimer();
            // TODO
            StreamCompaction::Common::kernMapToBoolean <<<blockNum, blockSize>>>
                (Num, dev_bools, dev_dataCopy);
            cudaDeviceSynchronize();
            //make a copy of bools data
            cudaMemcpy(dev_scan_Result, dev_bools, Num * sizeof(int), cudaMemcpyDeviceToDevice);

            //sweep up
            for (int i = 0; i <= ilog2ceil(Num); i++) {
                kernUpSweep << <blockNum, blockSize >> > (Num, pow(2, i + 1), dev_scan_Result);
            }
            cudaDeviceSynchronize();

            //set dev_scan_Result[n-1] = 0?
            cudaMemset(dev_scan_Result + Num - 1, 0, sizeof(int));

            //down-sweep
            for (int i = ilog2ceil(Num); i >= 0; i--) {
                kernDownSweep <<<blockNum, blockSize >> > (Num, pow(2, i + 1), dev_scan_Result);
            }
            cudaDeviceSynchronize();
            //scatter
            StreamCompaction::Common::kernScatter << <blockNum, blockSize >> >
                (Num, dev_ScatterResult, dev_dataCopy, dev_bools, dev_scan_Result);

            timer().endGpuTimer();

            //get num of result
            int num = 0;
            cudaMemcpy(&num, dev_scan_Result + Num - 1, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(odata, dev_ScatterResult, num * sizeof(int), cudaMemcpyDeviceToHost);

            
            cudaFree(dev_bools);
            cudaFree(dev_dataCopy);
            cudaFree(dev_scan_Result);
            cudaFree(dev_ScatterResult);
            
            return num;
        }

        __global__ void kernBittoBool(int n, int whichbit, int* idata, int* b, int* e)
        {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index >= n) return;
            int bit = (idata[index] >> whichbit) & 1;
            b[index] = bit;
            e[index] = !bit;
        }

        __global__ void kernComputeTarray(int n, int totalFalse, int* f, int* t)
        {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index >= n) return;
            t[index] = index - f[index] + totalFalse;
        }

        __global__ void kernComputeDarray(int n, int* b, int* t, int* f, int* d)
        {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index >= n) return;
            d[index] = b[index] ? t[index] : f[index];
        }

        __global__ void kernScatterToOutput(int n, int* d, int* idata, int* odata)
        {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index >= n) return;
            odata[d[index]] = idata[index];
        }

        void RadixSort(int n, int *odata, const int* idata )
        {
            int maxElement = *(std::max_element(idata, idata + n));
            int bitNum = ilog2ceil(maxElement);
            int* dev_idata;
            int* dev_b;
            int* dev_e;
            int* dev_f;
            int* dev_t;
            int* dev_d;
            int* dev_output;
            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            cudaMalloc((void**)&dev_b, n * sizeof(int));
            cudaMalloc((void**)&dev_e, n * sizeof(int));
            cudaMalloc((void**)&dev_f, n * sizeof(int));
            cudaMalloc((void**)&dev_t, n * sizeof(int));
            cudaMalloc((void**)&dev_d, n * sizeof(int));
            cudaMalloc((void**)&dev_output, n * sizeof(int));

            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            int blkSize = 128;
            int blkNum = (n + blkSize - 1) / blkSize;

            int totalFalse = 0;
            timer().startGpuTimer();
            for (int i = 0; i < bitNum; i++) {
                //create dev_b and dev_e
                kernBittoBool << <blkNum, blkSize >> > (n,i, dev_idata, dev_b, dev_e);
                //scan to create dev_f
                scanWithoutTimer(n, dev_f, dev_e);
                //compute totalFalse --> should use memcpy
                int e, f;
                cudaMemcpy(&e, dev_e + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
                cudaMemcpy(&f, dev_f + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
                totalFalse = e + f;
                //compute t array
                kernComputeTarray << < blkNum, blkSize >> > (n, totalFalse, dev_f, dev_t);
                //scatter
                kernComputeDarray << < blkNum, blkSize >> > (n, dev_b, dev_t, dev_f, dev_d);
                kernScatterToOutput << <blkNum, blkSize >> > (n, dev_d, dev_idata, dev_output);
                std::swap(dev_output, dev_idata);
            }
            timer().endGpuTimer();
            cudaMemcpy(odata, dev_idata, n * sizeof(int), cudaMemcpyDeviceToHost);
        }

    }
}
