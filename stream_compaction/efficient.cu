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

        // I think this is inefficient.
        __global__ void kernUpSweep(int* data, int n) {
            int idx = blockDim.x * blockIdx.x + threadIdx.x + 1;
            if (idx > n) {
                return;
            }
#pragma unroll
            for (int stride = 2; stride <= n; stride <<= 1) {
                if (idx % stride) {
                    data[idx - 1] += data[idx - stride / 2 - 1];
                }
                __syncthreads();
            }
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

        void devScanInPlace(int* devData, int size) {
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
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startGpuTimer();

            // TODO
            int size = ceilPow2(n);
            int* data;
            cudaMalloc(&data, size * sizeof(int));
            cudaMemcpy(data, idata, n * sizeof(int), cudaMemcpyKind::cudaMemcpyHostToDevice);
            cudaMemset(data + n, 0, (size - n) * sizeof(int));

            devScanInPlace(data, size);
            
            cudaMemcpy(odata, data, n * sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost);
            cudaFree(data);

            timer().endGpuTimer();
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
            timer().startGpuTimer();
            // TODO
            int* in, * out;
            int bytes = n * sizeof(int);
            cudaMalloc(&in, bytes);
            cudaMalloc(&out, bytes);
            cudaMemcpy(in, idata, bytes, cudaMemcpyKind::cudaMemcpyHostToDevice);

            int size = ceilPow2(n);
            int* indices;
            cudaMalloc(&indices, size * sizeof(int));
            cudaMemset(indices + n, 0, (size - n) * sizeof(int));

            int blockSize = Common::getDynamicBlockSizeEXT(n);
            int blockNum = (n + blockSize - 1) / blockSize;
            Common::kernMapToBoolean<<<blockNum, blockSize>>>(n, indices, in);

            devScanInPlace(indices, size);
            Common::kernScatter<<<blockNum, blockSize>>>(n, out, in, in, indices);

            int compactedSize;
            cudaMemcpy(&compactedSize, indices + n - 1, sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost);
            compactedSize += (idata[n - 1] != 0);

            cudaMemcpy(odata, out, compactedSize * sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost);

            cudaFree(indices);
            cudaFree(in);
            cudaFree(out);

            timer().endGpuTimer();
            return compactedSize;
        }
    }
}
