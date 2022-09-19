#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"
#include <device_launch_parameters.h>

#define USE_OPTIMIZATION 1

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernUpSweep(int n, int stride, int *odata)
        {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
#if USE_OPTIMIZATION
            index = stride * index + stride - 1;
#endif
            if (index >= n) return;

#if USE_OPTIMIZATION
            odata[index] += odata[index - stride / 2];
#else
            if ((index + 1) % stride == 0)
            {
                odata[index] += odata[index - stride / 2];
            }
#endif

        }

        __global__ void kernDownSweep(int n, int stride, int* odata)
        {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
#if USE_OPTIMIZATION
            index = stride * index + stride - 1;
#endif
            if (index >= n) return;

#if USE_OPTIMIZATION
            int leftChildVal = odata[index];
            odata[index] += odata[index - stride / 2];
            odata[index - stride / 2] = leftChildVal;
#else
            if ((index + 1) % stride == 0)
            {
                int leftChildVal = odata[index];
                odata[index] += odata[index - stride / 2];
                odata[index - stride / 2] = leftChildVal;
            }
#endif

        }

        void efficientScan(int n, int levelCount, int* dev_odata, int blockSize)
        {
            dim3 blockNum = (n + blockSize - 1) / blockSize;
            int stride = 1;
#if USE_OPTIMIZATION
            int sizeRequired = n;
#endif
            // Up-Sweep
            for (int d = 0; d < levelCount; ++d)
            {
#if USE_OPTIMIZATION
                sizeRequired /= 2;
                blockNum = (sizeRequired + blockSize - 1) / blockSize;
#endif
                stride *= 2;
                kernUpSweep<<<blockNum, blockSize>>>(n, stride, dev_odata);
            }

            // Down-Sweep
            cudaMemset(dev_odata + n - 1, 0, sizeof(int));
            for (int d = levelCount - 1; d >= 0; --d)
            {
#if USE_OPTIMIZATION
                sizeRequired *= 2;
                blockNum = (sizeRequired + blockSize - 1) / blockSize;
#endif
                kernDownSweep<<<blockNum, blockSize>>>(n, stride, dev_odata);
                stride /= 2;
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            int* dev_odata;
            int levelCount = ilog2ceil(n);
            int arraySize = 1 << levelCount;
            cudaMalloc((void**)&dev_odata, arraySize * sizeof(int));
            cudaMemcpy(dev_odata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            timer().startGpuTimer();
            // TODO
            int blockSize = 128;
            efficientScan(arraySize, levelCount, dev_odata, blockSize);

            timer().endGpuTimer();

            cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(dev_odata);
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
            int *dev_idata;
            int *dev_odata;
            int *dev_bools;
            int *dev_indices;
            int levelCount = ilog2ceil(n);
            int arraySize = 1 << levelCount;
            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            cudaMalloc((void**)&dev_odata, n * sizeof(int));
            cudaMalloc((void**)&dev_bools, n * sizeof(int));
            cudaMalloc((void**)&dev_indices, arraySize * sizeof(int));
            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            
            timer().startGpuTimer();
            // TODO
            int blockSize = 128;
            dim3 blockNum = (n + blockSize - 1) / blockSize;
            Common::kernMapToBoolean<<<blockNum, blockSize>>>(n, dev_bools, dev_idata);

            cudaMemcpy(dev_indices, dev_bools, n * sizeof(int), cudaMemcpyDeviceToDevice);
            cudaMemset(dev_indices + n, 0, (arraySize - n) * sizeof(int));
            efficientScan(arraySize, levelCount, dev_indices, blockSize);

            Common::kernScatter<<<blockNum, blockSize>>>(n, dev_odata, dev_idata, dev_bools, dev_indices);

            timer().endGpuTimer();
            cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);
            int elementCount = 0;
            cudaMemcpy(&elementCount, dev_indices + arraySize - 1, sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(dev_idata);
            cudaFree(dev_odata);
            cudaFree(dev_bools);
            cudaFree(dev_indices);
            return elementCount;
        }
    }
}
