#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

#include <device_launch_parameters.h>
#include <device_functions.h>

#define blockSize 128
#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        int* dev_idata;
        int* dev_odata;

        // TODO: __global__
        __global__ void kernScan(int n, int offset, int* odata, const int* idata) {
          int index = threadIdx.x + (blockIdx.x * blockDim.x);
          if (index >= n) {
            return;
          }

          if (index >= offset) {
            odata[index] = idata[index - offset] + idata[index];
          }
          else {
            odata[index] = idata[index];
          }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
            // TODO
            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            cudaMalloc((void**)&dev_odata, n * sizeof(int));
            checkCUDAErrorWithLine("cudaMalloc failed");

            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAErrorWithLine("memcpy idata failed!");

            dim3 fullBlocksPerGrid((n - blockSize + 1) / blockSize);

            int logOfN = ilog2ceil(n);

            for (int d = 1; d <= logOfN; ++d) {
              int offset = pow(2, d - 1);
              kernScan << < fullBlocksPerGrid, blockSize >> > (n, offset, dev_odata, dev_idata);

              // Needed to make sure we don't launch next iteration while prev is running
              cudaDeviceSynchronize();

              std::swap(dev_odata, dev_idata);
            }

            // Now result buffer is in dev_idata
            cudaMemcpy(odata, dev_idata, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAErrorWithLine("memcpy back to odata failed");

            //printf("mewo mewomoemomemeow");
            //for (int i = 0; i < n; ++i) {
            //  printf("%d ", odata[i]);
            //}

            cudaFree(dev_idata);
            cudaFree(dev_odata);
            timer().endGpuTimer();
        }
    }
}
