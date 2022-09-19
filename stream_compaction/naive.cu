#define GLM_FORCE_CUDA
#include <cuda.h>
#include <stdio.h>
#include <cmath>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_functions.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;

        #define blockSize 8

        int* dev_idata;
        int* dev_odata;
        int* dev_buf;

        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        // TODO: __global__

        __global__ void kernScan(int N, int* idata, int* odata, int depth) {
            int k = threadIdx.x + (blockIdx.x * blockDim.x);
            if (k >= N) {
                return;
            }

            if (k >= 1 << (depth - 1)) {
                odata[k] = idata[k - (1 << (depth - 1))] + idata[k];
            }
            else {
                odata[k] = idata[k];
            }
            
        }

        void zeroArray(int n, int* a) {
            for (int i = 0; i < n; i++) {
                a[i] = 0;
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startGpuTimer();

            int arrLen;
            int maxDepth = ilog2ceil(n);
            maxDepth > ilog2(n) ? arrLen = pow(2, maxDepth) : arrLen = n;
            zeroArray(arrLen, odata);

            dim3 threadsPerBlock(arrLen/blockSize);

            int* buf = new int[arrLen];

            for (int i = 0; i < arrLen; i++) {
                if (i < n) {
                    buf[i] = idata[i];
                }
                else {
                    buf[i] = 0;
                }
            }

            // TODO
            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            cudaMalloc((void**)&dev_odata, arrLen * sizeof(int));
            cudaMalloc((void**)&dev_buf, arrLen * sizeof(int));

            cudaMemcpy(dev_idata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
            cudaMemcpy(dev_odata, odata, sizeof(int) * arrLen, cudaMemcpyHostToDevice);
            cudaMemcpy(dev_buf, buf, sizeof(int) * arrLen, cudaMemcpyHostToDevice);

            for (int i = 1; i <= maxDepth; i++) {
                kernScan << <threadsPerBlock, blockSize >> > (arrLen, dev_buf, dev_odata, i);
                cudaDeviceSynchronize();
                cudaMemcpy(dev_buf, dev_odata, sizeof(int) * arrLen, cudaMemcpyDeviceToDevice);
            }         

            cudaMemcpy((void**)idata, dev_idata, sizeof(int) * n, cudaMemcpyDeviceToHost);
            cudaMemcpy(odata, dev_odata, sizeof(int) * arrLen, cudaMemcpyDeviceToHost);
            cudaMemcpy(buf, dev_buf, sizeof(int) * arrLen, cudaMemcpyDeviceToHost);

            cudaFree(dev_idata);
            cudaFree(dev_odata); 
            cudaFree(dev_buf);

            timer().endGpuTimer();
        }
    }
}
