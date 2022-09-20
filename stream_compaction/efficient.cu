#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"
#include <iostream>
#include <device_launch_parameters.h>


namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */

        __global__ void kernUpsweepStep(int n, int destStride, int srcStride, int *data) {
            int idx = threadIdx.x + (blockIdx.x * blockDim.x);
            int actualIdx = (idx + 1) * destStride - 1;
            if (actualIdx >= n) {
                return;
            }
            data[actualIdx] += data[actualIdx - srcStride];
        }

        __global__ void kernDownsweepStep(int n, int destStride, int srcStride, int* data) {
            int idx = threadIdx.x + (blockIdx.x * blockDim.x);
            int actualIdx = (idx + 1) * destStride - 1;
            if (actualIdx >= n) {
                return;
            }
            int temp = data[actualIdx - srcStride];
            data[actualIdx - srcStride] = data[actualIdx];
            data[actualIdx] += temp;
        }

        void scanWithoutTimer(int n, dim3 blocksPerGrid, int* dev_data) {
            // TODO

            for (int d = 0; d <= ilog2ceil(n); d++) {
                kernUpsweepStep << <blocksPerGrid, blockSize >> > (n, std::pow(2, d + 1), std::pow(2, d), dev_data);
                cudaDeviceSynchronize();
            }

            int zero = 0;
            cudaMemcpy(dev_data + n - 1, &zero, sizeof(int), cudaMemcpyHostToDevice);

            for (int d = ilog2ceil(n); d >= 0; d--) {
                kernDownsweepStep << <blocksPerGrid, blockSize >> > (n, std::pow(2, d + 1), std::pow(2, d), dev_data);
                cudaDeviceSynchronize();
            }
        }

        int closestPower(int num) {
            int i = 0;
            while (num > std::pow(2, i)) {
                i++;
            }
            return std::pow(2, i);
        }

        int* zeros(int num) {
            int *arr = (int*)malloc(num * sizeof(int));
            for (int i = 0; i < num; i++) {
                arr[i] = 0;
            }
            return arr;
        }

        void scan(int n, int *odata, const int *idata) {
            int nPot = closestPower(n);

            dim3 fullBlocksPerGrid((nPot + blockSize - 1) / blockSize);

            int* dev_data;

            cudaMalloc((void**)&dev_data, nPot * sizeof(int));
            checkCUDAError("Error during cudaMalloc dev_data");

            cudaMemcpy(dev_data + nPot - n, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("Error during cudaMemcpy idata ==> dev_data");

            int* zero = zeros(n);

            cudaMemcpy(dev_data, zero, (nPot - n) * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("Error during cudaMemcpy zero ==> dev_data");

            cudaDeviceSynchronize();

            timer().startGpuTimer();

            scanWithoutTimer(nPot, fullBlocksPerGrid, dev_data);
            
            timer().endGpuTimer();

            cudaMemcpy(odata, dev_data + nPot - n, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("Error during cudaMemcpy odata");

            cudaFree(dev_data);
            checkCUDAError("Error during cudaFree dev_data");

            free(zero);
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
            int nPot = closestPower(n);
            
            dim3 fullBlocksPerGrid((nPot + blockSize - 1) / blockSize);

            int* dev_idata, * dev_bools, * dev_indices, int* dev_odata;

            cudaMalloc((void**)&dev_idata, nPot * sizeof(int));
            checkCUDAError("Error during cudaMalloc dev_idata");

            cudaMalloc((void**)&dev_bools, nPot * sizeof(int));
            checkCUDAError("Error during cudaMalloc dev_bools");

            cudaMalloc((void**)&dev_indices, nPot * sizeof(int));
            checkCUDAError("Error during cudaMalloc dev_indices");

            cudaMalloc((void**)&dev_odata, nPot * sizeof(int));
            checkCUDAError("Error during cudaMalloc dev_odata");

            cudaMemcpy(dev_idata + nPot - n, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("Error during cudaMemcpy dev_data");

            int* zero = zeros(n);

            cudaMemcpy(dev_idata, zero, (nPot - n) * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("Error during cudaMemcpy zero ==> dev_data");

            cudaDeviceSynchronize();
            
            timer().startGpuTimer();
            //// TODO
            //

            StreamCompaction::Common::kernMapToBoolean << <fullBlocksPerGrid, blockSize >> > (nPot, dev_bools, dev_idata);

            cudaMemcpy(dev_indices, dev_bools, nPot * sizeof(int), cudaMemcpyDeviceToDevice);
            checkCUDAError("Error during cudaMemcpy dev_data");

            scanWithoutTimer(nPot, fullBlocksPerGrid, dev_indices);
            StreamCompaction::Common::kernScatter << <fullBlocksPerGrid, blockSize >> > (n, dev_odata + nPot - n, dev_idata + nPot - n, dev_bools + nPot - n, dev_indices + nPot - n);

            timer().endGpuTimer();

            cudaMemcpy(odata, dev_odata + nPot - n, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("Error during cudaMemcpy dev_odata");

            int count = 0;
            int lastbool = 0;
            cudaMemcpy(&count, dev_indices + nPot - 1, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&lastbool, dev_bools + nPot - 1, sizeof(int), cudaMemcpyDeviceToHost);

            count += lastbool;

            cudaFree(dev_odata);
            checkCUDAError("Error during cudaFree dev_odata");

            cudaFree(dev_indices);
            checkCUDAError("Error during cudaFree dev_indices");

            cudaFree(dev_bools);
            checkCUDAError("Error during cudaFree dev_bools");

            cudaFree(dev_idata);
            checkCUDAError("Error during cudaFree dev_idata");

            return count;
        }
    }
}
