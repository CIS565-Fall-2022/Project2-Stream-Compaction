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
        // TODO: __global__
        __global__ void kernScan(int n, int d, int* odata, const int* idata) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n) {
                return;
            }
            if (index >= (1 << (d - 1))) {
                odata[index] = idata[index - (1 << (d - 1))] + idata[index];
            }
            else {
                odata[index] = idata[index];
            }
        }

        __global__ void kernInclusiveToExclusive(int n, int* odata, const int* idata) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n) {
                return;
            }

            if (index == 0) {
                odata[index] = 0;
                return;
            }
            odata[index] = idata[index - 1];
        }


        ///**
        // * Performs prefix-sum (aka scan) on idata, storing the result into odata.
        // */
        void scan(int n, int *odata, const int *idata) {
            dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);
            
            // Put host arrays onto device
            int* dev_odata;
            int* dev_idata;
            cudaMalloc((void**)&dev_odata, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_odata failed!");
            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_idata failed!");
            cudaMemcpy(dev_odata, odata, n * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            cudaDeviceSynchronize();

            timer().startGpuTimer();
            for (int i = 1; i <= ilog2ceil(n); ++i) {
                kernScan << <fullBlocksPerGrid, blockSize >> > 
                        (n, i, dev_odata, dev_idata);
                checkCUDAError("kernScan failed!");
                cudaDeviceSynchronize();
                int* temp = dev_idata;
                dev_idata = dev_odata;
                dev_odata = temp;
            }
            cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);
            kernInclusiveToExclusive << <fullBlocksPerGrid, blockSize >> > (n, dev_odata, dev_idata);
            checkCUDAError("kernInclusiveToExclusive failed!");
            cudaDeviceSynchronize();
            timer().endGpuTimer();

            cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(dev_odata);
            cudaFree(dev_idata);
        }
    }
}
