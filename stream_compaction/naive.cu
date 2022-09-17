#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

#include <device_launch_parameters.h>
#include <device_functions.h>
#include <thrust/device_ptr.h>

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
        int* dev_unpadded_idata;

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

        // Pad the data with 1 zero at the beginning
        // And enough zeroes at the end
        // odata should be buffer of size paddedLength = the next power of two after and including (n + 1)
        __global__ void kernShiftAndPadInput(int n, int paddedLength, int* odata, int* idata) {
          int index = threadIdx.x + (blockIdx.x * blockDim.x);
          if (index == 0) {
            odata[index] = 0;
          }
          else if (index <= n) {
            odata[index] = idata[index - 1];
          }
          else if (index < paddedLength) {
            odata[index] = 0;
          }
          else {
            return;
          }
        }

        void printCudaArray(int n, int* dev_array) {
          int* tempArray = (int*) malloc(n * sizeof(int));
          cudaMemcpy(tempArray, dev_array, n * sizeof(int), cudaMemcpyDeviceToHost);
          printf("Print array -----------\n");
          for (int i = 0; i < n; ++i) {
            printf("%d ", tempArray[i]);
          }
          free(tempArray);
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
            // TODO
            
            // Add 1 because we're going to offset by a zero for exclusive scan
            int exponent = ilog2ceil(n + 1);
            int paddedLength = pow(2, exponent);
            // Input and output should be padded by 1s and 0s
            cudaMalloc((void**)&dev_unpadded_idata, n * sizeof(int));
            cudaMalloc((void**)&dev_idata, paddedLength * sizeof(int));
            cudaMalloc((void**)&dev_odata, paddedLength * sizeof(int));
            checkCUDAErrorWithLine("cudaMalloc failed");

            cudaMemcpy(dev_unpadded_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAErrorWithLine("memcpy idata failed!");

            dim3 fullBlocksPerGrid((paddedLength + blockSize - 1) / blockSize);

            kernShiftAndPadInput<<<fullBlocksPerGrid, blockSize>>>
              (n, paddedLength, dev_idata, dev_unpadded_idata);

            //printCudaArray(paddedLength, dev_idata);

            for (int d = 1; d <= exponent; ++d) {
              int offset = pow(2, d - 1);
              kernScan << < fullBlocksPerGrid, blockSize >> > (paddedLength, offset, dev_odata, dev_idata);

              // Needed to make sure we don't launch next iteration while prev is running
              cudaDeviceSynchronize();

              std::swap(dev_odata, dev_idata);
            }

            // Now result buffer is in dev_idata
            // We only need the first n elements of the total paddedLength elements
            cudaMemcpy(odata, dev_idata, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAErrorWithLine("memcpy back to odata failed");

            cudaFree(dev_idata);
            cudaFree(dev_odata);
            cudaFree(dev_unpadded_idata);
            timer().endGpuTimer();
        }
    }
}
