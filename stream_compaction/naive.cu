#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"
#include "iostream"

#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;

        /*! Block size used for CUDA kernel launch. */
        #define blockSize 128

        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        // TODO: __global__
        // offset calculated from depth
        __global__ void kernScanHelper(int n, int offset, int *odata, int *idata) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index < n) {
                if (index >= offset) {
                    int a = idata[index - offset];
                    int b = idata[index];
                    odata[index] = a + b;
                }
                else {
                     odata[index] = idata[index];
                }
            }
        }

        __global__ void kernPrependZero(int n, int* odata, int* idata) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index < n) {
                if (index == 0) {
                    odata[index] = 0;
                }
                else {
                    odata[index] = idata[index - 1];
                }
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            int fullBlocksPerArray = (n + blockSize - 1) / blockSize;

            timer().startGpuTimer();

            // empty buffer as odata1 and odata2
            int* dev_odata1;
            int* dev_odata2;

            cudaMalloc((void**)&dev_odata1, n * sizeof(int));
            checkCUDAErrorWithLine("cudaMalloc dev_odata1 failed!");

            cudaMalloc((void**)&dev_odata2, n * sizeof(int));
            checkCUDAErrorWithLine("cudaMalloc dev_odata2 failed!");

            // copy contents of idata into odata so we can just pass in odata.
            // set values of odata2 to 0s
            // exclusive scan will never set the 0th index
            cudaMemset(dev_odata2, 0, n * sizeof(int));
            checkCUDAErrorWithLine("cudaMemset dev_odata2 to 0'sfailed!");


            // memcpy contents of idata to odata1
            cudaMemcpy(dev_odata1, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAErrorWithLine("cudaMemcpy idata to dev_odata1 Host to Device failed!");

            // for depth from 1 to 2^d-1, for each k in parallel, invoke scan on an offset and odata.
            for (int depth = 0; depth < ilog2ceil(n); depth++) {
                int offset = pow(2, depth);

                kernScanHelper << <fullBlocksPerArray, blockSize>>>(n, offset, dev_odata2, dev_odata1);
                // wait for threads
                cudaDeviceSynchronize();

                std::swap(dev_odata2, dev_odata1);
            }

            // initialize new odataFinal with enough space with 1 in front
            int* dev_odataFinal;
            cudaMalloc((void**)&dev_odataFinal, n * sizeof(int));
            checkCUDAErrorWithLine("cudaMalloc dev_odataFinal failed!");

            // bit shift towards the right and add 0 to front of odata, put it in odataFinal
            // dev_odata1 should always have the most updated data.
            kernPrependZero << <fullBlocksPerArray, blockSize >> > (n, dev_odataFinal, dev_odata1);

            // memcpy back from odata1 to odata
            cudaMemcpy(odata, dev_odataFinal, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAErrorWithLine("cudaMemcpy dev_odataFinal to odata failed!");

            /*for (int i = 0; i < n; i++) {
                std::cout << odata[i] << std::endl;
            }*/

            cudaFree(dev_odata1);
            cudaFree(dev_odata2);
            cudaFree(dev_odataFinal);

            timer().endGpuTimer();

        }
    }
}
