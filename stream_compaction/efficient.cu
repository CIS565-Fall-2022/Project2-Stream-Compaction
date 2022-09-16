#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
    namespace Efficient {
        enum class ScanSource { Host, Device };

        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernPartialUpSweep(int* data, int n, int stride) {
            int idx = blockDim.x * blockIdx.x + threadIdx.x + 1;
            if (idx > n) {
                return;
            }
            int mappedIdx = idx * stride - 1;
            data[mappedIdx] += data[mappedIdx - stride / 2];
        }

        __global__ void kernPartialDownSweep(int* data, int n, int stride) {
            int idx = blockDim.x * blockIdx.x + threadIdx.x + 1;
            if (idx > n) {
                return;
            }
            int mappedIdx = idx * stride - 1;
            data[mappedIdx] += data[mappedIdx - stride / 2];
            data[mappedIdx - stride / 2] = data[mappedIdx] - data[mappedIdx - stride / 2];
        }

        __global__ void kernScanShared(int* data, int n) {
            extern __shared__ int shared[];

            int idx = threadIdx.x + 1;
            int globalIdx = blockDim.x * blockIdx.x + threadIdx.x;

            if (globalIdx > n) {
                return;
            }
            shared[idx - 1] = data[globalIdx];
            __syncthreads();

#pragma unroll
            for (int stride = 1, active = blockDim.x >> 1; stride < blockDim.x; stride <<= 1, active >>= 1) {
                if (idx <= active) {
                    int mappedIdx = idx * stride * 2 - 1;
                    shared[mappedIdx] += shared[mappedIdx - stride];
                }
                __syncthreads();
            }

            if (idx == 1) {
                shared[blockDim.x - 1] = 0;
            }
            __syncthreads();

#pragma unroll
            for (int stride = blockDim.x >> 1, active = 1; stride >= 1; stride >>= 1, active <<= 1) {
                if (idx <= active) {
                    int mappedIdx = idx * stride * 2 - 1;
                    shared[mappedIdx] += shared[mappedIdx - stride];
                    shared[mappedIdx - stride] = shared[mappedIdx] - shared[mappedIdx - stride];
                }
                __syncthreads();
            }
            data[globalIdx] = shared[idx - 1];
        }

        void devScanInPlace(int* devData, int size) {
            if (size != ceilPow2(size)) {
                throw std::runtime_error("devScanInPlace:: size not pow of 2");
            }

            for (int stride = 2; stride <= size; stride <<= 1) {
                int num = size / stride;
                int blockSize = Common::getDynamicBlockSizeEXT(num);
                int blockNum = (num + blockSize - 1) / blockSize;
                kernPartialUpSweep<<<blockNum, blockSize>>>(devData, num, stride);
            }

            cudaMemset(devData + size - 1, 0, sizeof(int));
            for (int stride = size; stride >= 2; stride >>= 1) {
                int num = size / stride;
                int blockSize = Common::getDynamicBlockSizeEXT(num);
                int blockNum = (num + blockSize - 1) / blockSize;
                kernPartialDownSweep<<<blockNum, blockSize>>>(devData, num, stride);
            }
        }

        /**
         * Performs scan on every block
         * Note: results of blocks are partial, need to be merged
         */
        void devScanInPlaceShared(int* devData, int size, int blockSize) {
            if (size != ceilPow2(size)) {
                throw std::runtime_error("devScanInPlace:: size not pow of 2");
            }
            int blockNum = (size + blockSize - 1) / blockSize;
            kernScanShared<<<blockNum, blockSize>>>(devData, size);
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            // TODO
            int size = ceilPow2(n);
            int* data;
            cudaMalloc(&data, size * sizeof(int));
            cudaMemcpy(data, idata, n * sizeof(int), cudaMemcpyKind::cudaMemcpyHostToDevice);

            timer().startGpuTimer();

            devScanInPlace(data, size);

            timer().endGpuTimer();
            
            cudaMemcpy(odata, data, n * sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost);
            cudaFree(data);
        }

        void scanWithSharedMemory(int* out, const int* in, int n) {
            int size = ceilPow2(n);
            int* data;
            cudaMalloc(&data, size * sizeof(int));
            cudaMemcpy(data, in, n * sizeof(int), cudaMemcpyKind::cudaMemcpyHostToDevice);

            timer().startGpuTimer();

            int blockSize = 64;
            devScanInPlaceShared(data, size, blockSize);

            timer().endGpuTimer();

            cudaMemcpy(out, data, n * sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost);
            cudaFree(data);
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
            // TODO
            int* in, * out;
            int bytes = n * sizeof(int);
            cudaMalloc(&in, bytes);
            cudaMalloc(&out, bytes);
            cudaMemcpy(in, idata, bytes, cudaMemcpyKind::cudaMemcpyHostToDevice);

            int size = ceilPow2(n);
            int* indices;
            cudaMalloc(&indices, size * sizeof(int));

            timer().startGpuTimer();

            int blockSize = Common::getDynamicBlockSizeEXT(n);
            int blockNum = (n + blockSize - 1) / blockSize;

            Common::kernMapToBoolean<<<blockNum, blockSize>>>(n, indices, in);
            devScanInPlace(indices, size);
            Common::kernScatter<<<blockNum, blockSize>>>(n, out, in, in, indices);

            timer().endGpuTimer();

            int compactedSize;
            cudaMemcpy(&compactedSize, indices + n - 1, sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost);
            compactedSize += (idata[n - 1] != 0);

            cudaMemcpy(odata, out, compactedSize * sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost);

            cudaFree(indices);
            cudaFree(in);
            cudaFree(out);

            return compactedSize;
        }
    }
}
