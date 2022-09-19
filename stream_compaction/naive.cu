#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

#include "device_launch_parameters.h"

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        // TODO: __global__
        __global__ void kernNaiveScan(int d, int n, int* odata, int* idata)
        {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index >= n) return;
            int pow2 = 1 << (d -1);
            if (index >= pow2) {
                odata[index] = idata[index - pow2] + idata[index];
            }
            else {
                odata[index] = idata[index];
            }
        }

        __host__ bool isPowerof2(int n) {
            return (n & (n - 1)) == 0;
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            int* dev_array1 = nullptr;
            int* dev_array2 = nullptr;
            int Num = 1 << ilog2ceil(n);

            cudaMalloc((void**)&dev_array1, Num * sizeof(int));
            checkCUDAErrorFn("malloc dev_array1 failed.");
            cudaMalloc((void**)&dev_array2, Num * sizeof(int));
            checkCUDAErrorFn("malloc dev_array2 failed.");

            //cudaMalloc((void**)&dev_array1, n * sizeof(int));
            //checkCUDAErrorFn("malloc dev_array1 failed.");
            //cudaMalloc((void**)&dev_array2, n * sizeof(int));
            //checkCUDAErrorFn("malloc dev_array2 failed.");

            cudaMemset(dev_array1, 0, Num * sizeof(int));
            checkCUDAErrorFn("memset dev_array1 to 0 failed.");
            cudaMemset(dev_array2, 0, Num * sizeof(int));
            checkCUDAErrorFn("memset dev_array2 to 0 failed.");

            cudaMemcpy(dev_array1, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAErrorFn("memcopy idata to dev_array1 failed.");

            int blocksize = 256;
            int blockCount = (n + blocksize - 1) / blocksize;

            

            timer().startGpuTimer();
            // TODO
            for (int i = 1; i <= ilog2ceil(Num); i++) {
                kernNaiveScan <<<blockCount, blocksize >>> (i, Num, dev_array2, dev_array1);
                cudaDeviceSynchronize();
                std::swap(dev_array1, dev_array2);
            }

            timer().endGpuTimer();
            cudaMemcpy(odata+1, dev_array1, (n-1) * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAErrorFn("memcopy dev_array1 to odata+1 failed.");
            odata[0] = 0;
            cudaFree(dev_array1);
            checkCUDAErrorFn("free dev_array1 failed.");
            cudaFree(dev_array2);
            checkCUDAErrorFn("free dev_array2 failed.");
        }
    }
}
