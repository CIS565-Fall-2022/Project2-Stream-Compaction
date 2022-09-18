#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }


        __global__ void kernUpSweep(int n, int exp2d,int exp2d1, int* idata) 
        {
            /*int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n)
            {
                return;
            }*/

            int index = threadIdx.x;
            int blockOffset = blockIdx.x * blockDim.x;

            if (index % exp2d1 == 0) {
                idata[index + blockOffset + exp2d1 - 1] += idata[index + blockOffset + exp2d - 1];
            }
        }

        __global__ void kernDownSweep(int n, int exp2d, int exp2d1, int* idata) {
            /*int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n)
            {
                return;
            }*/

            int index = threadIdx.x;
            int blockOffset = blockIdx.x * blockDim.x;

            if (index % exp2d1 == 0) {
                int t = idata[index + blockOffset+ exp2d - 1];
                idata[index + blockOffset + exp2d - 1] = idata[index + blockOffset + exp2d1 - 1];
                idata[index + blockOffset + exp2d1 - 1] += t;
            }
        }

        __global__ void kernCombine(int blockNum, int* original_buffer, int* inter_buffer)
        {
            int blockIndex = blockIdx.x;
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (blockIndex >= blockNum)
            {
                return;
            }

            original_buffer[index] += inter_buffer[blockIndex];
        }

        __global__ void kernSetZero(int blockNum, int size, int* idata)
        {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= blockNum)
            {
                return;
            }
            idata[(index + 1) * size - 1] = 0;
        }

        __global__ void kernSetInterZero(int n, int* idata)
        {
            idata[n - 1] = 0;
        }

        __global__ void kernPadZero(int n, int size, int* idata)
        {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= size - n)
            {
                return;
            }
            idata[index + n] = 0;
        }

        __global__ void kernInitInterBuffer(int blockNum, int size, int* original_buffer, int* inter_buffer)
        {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= blockNum)
            {
                return;
            }
            inter_buffer[index] = original_buffer[(index + 1) * size - 1];
        }

        __global__ void kernShiftLeft(int n, const int* idata, int* odata)
        {
            // exclusive to inclusive
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n)
            {
                return;
            }
            odata[index] += idata[index];
        }

        __global__ void kernShiftRight(int n, const int* idata, int* odata)
        {
            // inclusive to exclusive
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n)
            {
                return;
            }
            odata[index] -= idata[index];
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            int blockNum = (n + blockSize - 1) / blockSize;

            // we need to ensure that the block number needs to be smaller than the size of a single thread block
            if (blockNum > blockSize)
            {
                printf("blocksize too small, block number is larger than blockSize");
                return;
            }


            dim3 fullBlocksPerGrid(blockNum);
            int* input_buffer;
            int* dev_buffer;
            int bufferSize = blockNum * blockSize;

            cudaMalloc((void**)&dev_buffer, bufferSize * sizeof(int));
            cudaMemcpy(dev_buffer, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            cudaMalloc((void**)&input_buffer, n * sizeof(int));
            cudaMemcpy(input_buffer, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            // used for combine scanned blocks
            int* dev_inter_buffer;
            cudaMalloc((void**)&dev_inter_buffer, blockSize * sizeof(int));
            kernPadZero << <1, blockSize >> > (0, blockSize, dev_inter_buffer);

            timer().startGpuTimer();

            if (n != bufferSize)
            {
                // it is an arbitary sized array, we need to pad it with zero for the last block
                kernPadZero << <1, blockSize >> > (n, bufferSize, dev_buffer);
            }

            // we do scan for each block
            // up sweep
            for (int d = 0; d <= ilog2ceil(blockSize) - 1; ++d) {
                int exp2d = 1 << d;
                int exp2d1 = 1 << (d + 1);
                kernUpSweep << <fullBlocksPerGrid, blockSize >> > (n, exp2d,exp2d1, dev_buffer);
            }
            // down sweep

            // zero out the last element for each block
            dim3 numBlocksForBlock((blockNum + blockSize - 1) / blockSize);
            kernSetZero << <numBlocksForBlock, blockSize >> > (blockNum, blockSize, dev_buffer);

            for (int d = ilog2ceil(blockSize) - 1; d >= 0; --d) {
                int exp2d = 1 << d;
                int exp2d1 = 1 << (d + 1);
                kernDownSweep << <fullBlocksPerGrid, blockSize >> > (n, exp2d, exp2d1, dev_buffer);
            }

            // make them all inclusive
            kernShiftLeft << <fullBlocksPerGrid, blockSize >> > (n, input_buffer, dev_buffer);

            //now we have all those scanned blocks, we combine them into one
            kernInitInterBuffer<<<numBlocksForBlock , blockSize >>>(blockNum, blockSize, dev_buffer, dev_inter_buffer);
            // do a scan on this intermediate one also
            for (int d = 0; d <= ilog2ceil(blockSize) - 1; ++d) {
                int exp2d = 1 << d;
                int exp2d1 = 1 << (d + 1);
                kernUpSweep << <1, blockSize >> > (blockNum, exp2d, exp2d1, dev_inter_buffer);
            }
            kernSetInterZero << <1, 1 >> > (blockSize, dev_inter_buffer);
            for (int d = ilog2ceil(blockSize) - 1; d >= 0; --d) {
                int exp2d = 1 << d;
                int exp2d1 = 1 << (d + 1);
                kernDownSweep << <1, blockSize >> > (blockNum, exp2d, exp2d1, dev_inter_buffer);
            }

            // now we add those offsets to the original buffer
            kernCombine << <fullBlocksPerGrid, blockSize >> > (blockNum, dev_buffer, dev_inter_buffer);

            // make them all exclusive
            kernShiftRight<< <fullBlocksPerGrid, blockSize >> > (n, input_buffer, dev_buffer);

            timer().endGpuTimer();
            cudaMemcpy(odata, dev_buffer, n*sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(dev_buffer);
            cudaFree(input_buffer);
            cudaFree(dev_inter_buffer);
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

            dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);
            int* inputData;
            int* boolData;
            int* scanData;
            int* outputData;

            cudaMalloc((void**)&inputData, n * sizeof(int));
            cudaMemcpy(inputData, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            cudaMalloc((void**)&boolData, n * sizeof(int));
            cudaMalloc((void**)&scanData, n * sizeof(int));
            cudaMalloc((void**)&outputData, n * sizeof(int));
            timer().startGpuTimer();
            // TODO
            StreamCompaction::Common::kernMapToBoolean << <fullBlocksPerGrid, blockSize >> > (n, boolData, inputData);
            efficientScan(n, scanData, boolData);
            StreamCompaction::Common::kernScatter << <fullBlocksPerGrid, blockSize >> > (n, outputData, inputData, boolData, scanData);

            timer().endGpuTimer();

            int result = 0;
            int lastElement = 0;
            cudaMemcpy(&result, scanData + (n - 1), sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&lastElement, boolData + (n - 1), sizeof(int), cudaMemcpyDeviceToHost);
            result += lastElement;
            cudaMemcpy(odata, outputData, result * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(boolData);
            cudaFree(scanData);
            cudaFree(outputData);
            return result;
        }



        void efficientScan(int n, int* odata, const int* idata)
        {
            int blockNum = (n + blockSize - 1) / blockSize;

            // we need to ensure that the block number needs to be smaller than the size of a single thread block
            if (blockNum > blockSize)
            {
                printf("blocksize too small, block number is larger than blockSize");
                return;
            }


            dim3 fullBlocksPerGrid(blockNum);
            int* input_buffer;
            int* dev_buffer;
            int bufferSize = blockNum * blockSize;

            cudaMalloc((void**)&dev_buffer, bufferSize * sizeof(int));
            cudaMemcpy(dev_buffer, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            cudaMalloc((void**)&input_buffer, n * sizeof(int));
            cudaMemcpy(input_buffer, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            // used for combine scanned blocks
            int* dev_inter_buffer;
            cudaMalloc((void**)&dev_inter_buffer, blockSize * sizeof(int));
            kernPadZero << <1, blockSize >> > (0, blockSize, dev_inter_buffer);

            if (n != bufferSize)
            {
                // it is an arbitary sized array, we need to pad it with zero for the last block
                kernPadZero << <1, blockSize >> > (n, bufferSize, dev_buffer);
            }

            // we do scan for each block
            // up sweep
            for (int d = 0; d <= ilog2ceil(blockSize) - 1; ++d) {
                int exp2d = 1 << d;
                int exp2d1 = 1 << (d + 1);
                kernUpSweep << <fullBlocksPerGrid, blockSize >> > (n, exp2d, exp2d1, dev_buffer);
            }
            // down sweep

            // zero out the last element for each block
            dim3 numBlocksForBlock((blockNum + blockSize - 1) / blockSize);
            kernSetZero << <numBlocksForBlock, blockSize >> > (blockNum, blockSize, dev_buffer);

            for (int d = ilog2ceil(blockSize) - 1; d >= 0; --d) {
                int exp2d = 1 << d;
                int exp2d1 = 1 << (d + 1);
                kernDownSweep << <fullBlocksPerGrid, blockSize >> > (n, exp2d, exp2d1, dev_buffer);
            }

            // make them all inclusive
            kernShiftLeft << <fullBlocksPerGrid, blockSize >> > (n, input_buffer, dev_buffer);

            //now we have all those scanned blocks, we combine them into one
            kernInitInterBuffer << <numBlocksForBlock, blockSize >> > (blockNum, blockSize, dev_buffer, dev_inter_buffer);
            // do a scan on this intermediate one also
            for (int d = 0; d <= ilog2ceil(blockSize) - 1; ++d) {
                int exp2d = 1 << d;
                int exp2d1 = 1 << (d + 1);
                kernUpSweep << <1, blockSize >> > (blockNum, exp2d, exp2d1, dev_inter_buffer);
            }
            kernSetInterZero << <1, 1 >> > (blockSize, dev_inter_buffer);
            for (int d = ilog2ceil(blockSize) - 1; d >= 0; --d) {
                int exp2d = 1 << d;
                int exp2d1 = 1 << (d + 1);
                kernDownSweep << <1, blockSize >> > (blockNum, exp2d, exp2d1, dev_inter_buffer);
            }

            // now we add those offsets to the original buffer
            kernCombine << <fullBlocksPerGrid, blockSize >> > (blockNum, dev_buffer, dev_inter_buffer);

            // make them all exclusive
            kernShiftRight << <fullBlocksPerGrid, blockSize >> > (n, input_buffer, dev_buffer);

            cudaMemcpy(odata, dev_buffer, n * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(dev_buffer);
            cudaFree(input_buffer);
            cudaFree(dev_inter_buffer);
        }
    }
}
