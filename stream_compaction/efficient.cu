#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

#define blockSize 256

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernUpSweep(int n, int* data, int depth) {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index >= n) {
                return;
            }
            int offset = pow(2, depth + 1);
            if (index % offset == 0) {
                data[index + offset - 1] += data[index + (int)pow(2, depth) - 1];
            }
        }

        __global__ void kernDownSweep(int n, int* data, int depth) {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index >= n) {
                return;
            }
            int offset1 = pow(2, depth);
            int offset2 = pow(2, depth + 1);
            if (index % offset2 == 0) {
                int t = data[index + offset1 - 1];
                data[index + offset1 - 1] = data[index + offset2 - 1];
                data[index + offset2 - 1] += t;
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {       
            int power = ilog2ceil(n);
            int arraySize = pow(2, power);
            dim3 blockPerGrid((arraySize + blockSize - 1) / blockSize);
            dim3 threadPerBlock(blockSize);

            int* dev_data;

            // create memory
            cudaMalloc((void**)&dev_data, arraySize * sizeof(int));
            // set data and then copy the original data 
            cudaMemset(dev_data, 0, arraySize * sizeof(int));
            cudaMemcpy(dev_data, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            timer().startGpuTimer();
            // TODO
            for (int i = 0; i < power; ++i) {
                kernUpSweep << <blockPerGrid, threadPerBlock >> > (arraySize, dev_data, i);
            }
            // set the root to 0
            cudaMemset(dev_data + arraySize - 1, 0, sizeof(int));
            for (int i = power - 1; i >= 0; --i) {
                kernDownSweep << <blockPerGrid, threadPerBlock >> > (arraySize, dev_data, i);
            }
            timer().endGpuTimer();
            cudaMemcpy(odata, dev_data, n * sizeof(int), cudaMemcpyDeviceToHost);

            // free memory
            cudaFree(dev_data);
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
            int power = ilog2ceil(n);
            int arraySize = pow(2, power);
            dim3 blockPerGrid((arraySize + blockSize - 1) / blockSize);
            dim3 threadPerBlock(blockSize);

            int* dev_idata;
            int* dev_odata;
            int* dev_boolBuffer;
            int* dev_scanResultBuffer;

            // malloc
            cudaMalloc((void**)&dev_idata, arraySize * sizeof(int));
            cudaMalloc((void**)&dev_odata, arraySize * sizeof(int));
            cudaMalloc((void**)&dev_boolBuffer, arraySize * sizeof(int));
            cudaMalloc((void**)&dev_scanResultBuffer, arraySize * sizeof(int));

            // set data and copy data
            // important for non power of two data!
            // if not set to 0, when map to boolean, the extra data which is not 0 will cause damage
            cudaMemset(dev_idata , 0, arraySize * sizeof(int)); 
            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            timer().startGpuTimer();
            // TODO
            StreamCompaction::Common::kernMapToBoolean << <blockPerGrid, threadPerBlock>> > (arraySize, 
                                                                                                                                                                              dev_boolBuffer,
                                                                                                                                                                              dev_idata);
            cudaMemcpy(dev_scanResultBuffer, dev_boolBuffer, arraySize * sizeof(int), cudaMemcpyDeviceToDevice);

            for (int i = 0; i < power; ++i) {
                kernUpSweep << <blockPerGrid, threadPerBlock >> > (arraySize, dev_scanResultBuffer, i);
            }
            // set the root to 0
            cudaMemset(dev_scanResultBuffer + arraySize - 1, 0, sizeof(int));
            for (int i = power - 1; i >= 0; --i) {
                kernDownSweep << <blockPerGrid, threadPerBlock >> > (arraySize, dev_scanResultBuffer, i);
            }

            StreamCompaction::Common::kernScatter << <blockPerGrid, threadPerBlock >> > (arraySize,
                                                                                                                                                                 dev_odata, dev_idata,
                                                                                                                                                                 dev_boolBuffer, dev_scanResultBuffer);
            timer().endGpuTimer();

            int* host_scanResultBuffer = new int[arraySize];
            cudaMemcpy(host_scanResultBuffer, dev_scanResultBuffer, arraySize * sizeof(int), cudaMemcpyDeviceToHost);

            int resultCount = host_scanResultBuffer[arraySize - 1];
            cudaMemcpy(odata, dev_odata, resultCount * sizeof(int), cudaMemcpyDeviceToHost);
            return resultCount;
        }
    }
}
