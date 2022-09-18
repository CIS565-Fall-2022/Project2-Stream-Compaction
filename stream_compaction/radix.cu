#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"
#include "radix.h"
#include <iostream>
#include <thrust/host_vector.h>
#include <thrust/sort.h>

#define blockSize 128
namespace StreamCompaction {
	namespace Radix {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernCalBool(const int n, const int digit, int* odata, const int* idata) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n) {
                return;
            }
            int num = idata[index] >> digit;
            odata[index] = num & 1 ? 0 : 1;
        }

        __global__ void kernCalT(int n, int totalFalse, int* splitScan, int* odata) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n) {
                return;
            }
            odata[index] = index + totalFalse - splitScan[index];
        }

        __global__ void kernCalD(int n, int digit, int* odata, const int* arrayT, const int* splitScan, const int* splitBool) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n) {
                return;
            }
            int num = 1 - splitBool[index];
            //int num = ((idata[index] >> digit) & 1) ? 1 : 0;
            odata[index] = num ? arrayT[index] : splitScan[index];
        }

        __global__ void kernCalScatter(int n, int* odata, const int* idata, int* arrayD) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n) {
                return;
            }
            odata[arrayD[index]] = idata[index];

        }

        void thrustSort(int n, int* odata, const int* idata) {
            thrust::host_vector<int>dev_out(idata, idata + n);
            timer().startGpuTimer();
            thrust::sort(dev_out.begin(), dev_out.end());
            thrust::copy(dev_out.begin(), dev_out.end(), odata);
            timer().endGpuTimer();
        }

        void radixSort(int n, int* odata, const int* idata) {
            int maxBits = 0;
            for (int i = 0; i < n; i++) {
                int bits = 1;
                for (int k = idata[i]; k; k >>= 1) {
                    bits++;
                }
                maxBits = maxBits > bits ? maxBits : bits;
            }
            int maxN = 1 << ilog2ceil(n);
            dim3 blockDim((n + blockSize - 1) / blockSize);
            int* dev_in;
            int* dev_out;
            int* splitBool;//stores the bool result
            int* splitScan = nullptr; //then scan the 1s in splitBool
            int* arrayT;
            int* arrayD;//simply use the variable name from GPU Gem
            int totalFalse; 
            cudaMalloc((void**)&dev_in, n * sizeof(int));
            cudaMalloc((void**)&dev_out, n * sizeof(int));
            cudaMalloc((void**)&splitBool, n * sizeof(int));
            cudaMalloc((void**)&arrayT, n * sizeof(int));
            cudaMalloc((void**)&arrayD, n * sizeof(int));
            
            cudaMemcpy(dev_in, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            timer().startGpuTimer();
            for (int i = 0; i < maxBits; i++) {
                kernCalBool<<<blockDim, blockSize>>>(n, i, splitBool, dev_in);
                checkCUDAError("kernCalBool failed");
                cudaDeviceSynchronize();
                //Scan
                Efficient::fillArray(&splitScan, splitBool, maxN, n);
                Efficient::scanImpl(maxN, splitScan);
                int lastOfScan;
                cudaMemcpy(&lastOfScan, splitScan + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
                int lastOfBool;
                cudaMemcpy(&lastOfBool, splitBool + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
                totalFalse = lastOfScan + lastOfBool;
                kernCalT << <blockDim, blockSize >> > (n, totalFalse, splitScan, arrayT);
                checkCUDAError("kernCalT failed");
                cudaDeviceSynchronize();
                kernCalD << <blockDim, blockSize >> > (n, i, arrayD, arrayT, splitScan, splitBool);
                checkCUDAError("kernCalD failed");
                cudaDeviceSynchronize();
                kernCalScatter << <blockDim, blockSize >> > (n, dev_out, dev_in, arrayD);
                checkCUDAError("kernCalScatter failed");
                cudaDeviceSynchronize();

                //std::swap(dev_out, dev_in);
                cudaMemcpy(dev_in, dev_out, n * sizeof(int), cudaMemcpyDeviceToDevice);
            }
            timer().endGpuTimer();
            cudaMemcpy(odata, dev_out, n * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(dev_in); cudaFree(dev_out); cudaFree(splitBool); cudaFree(splitScan); cudaFree(arrayT); cudaFree(arrayD);
        }
	}
}