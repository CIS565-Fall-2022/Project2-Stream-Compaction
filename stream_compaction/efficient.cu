#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"
#include "common.cu"
using namespace StreamCompaction::Common;

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
            //Down sweep 
        __global__ void kernUpSweep(int* g_idata, int n, int d) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n) {
                return;
            }
            int pow2dplus1 = 1 << d + 1;
            int pow2d = 1 << d; // d0 p1, d1 p2
            if (index % pow2dplus1 == 0) {
                g_idata[index + pow2dplus1 - 1] += g_idata[index + pow2d - 1]; // 
            }
            //if (index % (1 << (d + 1)) == 0) {
            //    g_idata[index + (1 << (d + 1)) - 1] += g_idata[index + (1 << d) - 1];
            //}
        }
        __global__ void kernDownSweep(int* g_idata, int n, int d) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n) {
                return;
            }

            //if (index % (1 << (d + 1)) == 0) {
            //    int t = g_idata[index + (1 << d) - 1];
            //    g_idata[index + (1 << d) - 1] = g_idata[index + (1 << (d + 1)) - 1];
            //    g_idata[index + (1 << (d + 1)) - 1] += t;
            //}
            int pow2dplus1 = 1 << d + 1;
            int pow2d = 1 << d; // d0 p1, d1 p2

            if (index % pow2dplus1 == 0) {
                int temp = g_idata[index + pow2d - 1];
                g_idata[index + pow2d - 1] = g_idata[index + pow2dplus1 - 1];
                g_idata[index + pow2dplus1 - 1] += temp;  
            }
         }
        __global__ void kernScan(const int* idata, int* odata, int n, int d) {

        }

        __global__ void kernSetZero(int n, int* idata) {
            idata[n - 1] = 0;
        }

        /*__global__ void kernUpSweep(int d, int n, int* idata) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n) {
                return;
            }
            if (index % (1 << (d + 1)) == 0) {
                idata[index + (1 << (d + 1)) - 1] += idata[index + (1 << d) - 1];
            }
        }


        __global__ void kernDownSweep(int d, int n, int* idata) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n) {
                return;
            }
            if (index % (1 << (d + 1)) == 0) {
                int t = idata[index + (1 << d) - 1];
                idata[index + (1 << d) - 1] = idata[index + (1 << (d + 1)) - 1];
                idata[index + (1 << (d + 1)) - 1] += t;
            }
        }*/



        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {

            // allocate memory
            int paddedSize = 1 << ilog2ceil(n);
            dim3 blocksPerGrid((paddedSize + BLOCK_SIZE - 1) / BLOCK_SIZE);

            int* dev_in;
            cudaMalloc((void**)&dev_in, paddedSize * sizeof(int));
            cudaMemcpy(dev_in, idata, sizeof(int) * n, cudaMemcpyHostToDevice);

            timer().startGpuTimer();
            
            // Up Sweep Phase
            for (int d = 0; d <= ilog2ceil(paddedSize) - 1; d++) {
                kernUpSweep <<< blocksPerGrid, BLOCK_SIZE >>> ( dev_in, paddedSize, d);
                checkCUDAError("kernUpSweep failed");

            }
            kernSetZero << < 1, 1 >> > (paddedSize, dev_in);

            // Down Sweep Phase
            for (int d = ilog2ceil(paddedSize) - 1; d >= 0; d--) {
                kernDownSweep <<< blocksPerGrid, BLOCK_SIZE >>> (dev_in, paddedSize, d);
                checkCUDAError("kernDownSweep failed");

            }
            timer().endGpuTimer();
            
            // send the data to host 
            cudaMemcpy(odata, dev_in, sizeof(int) * (n), cudaMemcpyDeviceToHost);
            cudaFree(dev_in);

        }
        void scanNoTimer(int n, int* odata, const int* idata) {

            // allocate memory
            int paddedSize = 1 << ilog2ceil(n);
            dim3 blocksPerGrid((paddedSize + BLOCK_SIZE - 1) / BLOCK_SIZE);

            int* dev_in;
            cudaMalloc((void**)&dev_in, paddedSize * sizeof(int));
            cudaMemcpy(dev_in, idata, sizeof(int) * n, cudaMemcpyHostToDevice);

            // Up Sweep Phase
            for (int d = 0; d <= ilog2ceil(paddedSize) - 1; d++) {
                kernUpSweep << < blocksPerGrid, BLOCK_SIZE >> > (dev_in, paddedSize, d);
                checkCUDAError("kernUpSweep failed");

            }
            kernSetZero << < 1, 1 >> > (paddedSize, dev_in);

            // Down Sweep Phase
            for (int d = ilog2ceil(paddedSize) - 1; d >= 0; d--) {
                kernDownSweep << < blocksPerGrid, BLOCK_SIZE >> > (dev_in, paddedSize, d);
                checkCUDAError("kernDownSweep failed");

            }

            // send the data to host 
            cudaMemcpy(odata, dev_in, sizeof(int) * (n), cudaMemcpyDeviceToHost);
            cudaFree(dev_in);

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
            dim3 blocksPerGrid((n + BLOCK_SIZE - 1) / BLOCK_SIZE);

            int* count = new int[2];

            int* dev_in;
            int* dev_bool;
            int* dev_ScanRes;
            int* dev_out;
            cudaMalloc((void**)&dev_in, n * sizeof(int));
            cudaMalloc((void**)&dev_bool, n * sizeof(int));
            cudaMalloc((void**)&dev_ScanRes, n * sizeof(int));
            cudaMalloc((void**)&dev_out, n * sizeof(int));

            cudaMemcpy(dev_in, idata, sizeof(int) * n, cudaMemcpyHostToDevice);

            timer().startGpuTimer();
            // TODO
            kernMapToBoolean << <blocksPerGrid, BLOCK_SIZE >> > (n, dev_bool, dev_in);
            scanNoTimer(n, dev_ScanRes, dev_bool);
            kernScatter << <blocksPerGrid, BLOCK_SIZE >> > (n, dev_out, dev_in, dev_bool, dev_ScanRes);
            timer().endGpuTimer();

            cudaMemcpy(count, &dev_bool[n - 1], sizeof(int), cudaMemcpyDeviceToHost);
            //cudaMemcpy(count + 1, dev_ScanRes + n - 1, sizeof(int), cudaMemcpyDeviceToHost);

            //size equals to last of boolean array and last of boolean prefix sum array
            int size;
            cudaMemcpy(&size, &dev_ScanRes[n - 1], sizeof(int), cudaMemcpyDeviceToHost); // copy the last element of scan result
            size += count[0];

            cudaMemcpy(odata, dev_out, sizeof(int) * size, cudaMemcpyDeviceToHost);

            cudaFree(dev_in);
            cudaFree(dev_bool);
            cudaFree(dev_ScanRes);
            cudaFree(dev_out);
            return size;
        }
    }
}
