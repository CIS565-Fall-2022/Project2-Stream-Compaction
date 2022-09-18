#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

#define SCAN_EFFI_REDUCE_BANK_CONFLICT 1

namespace StreamCompaction {
    namespace Efficient {
        enum class ScanSource { Host, Device };

        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __device__ inline int bankOffset(int idx) {
            return idx >> 5;
        }
        __device__ inline int offsetAddr(int idx) {
#if SCAN_EFFI_REDUCE_BANK_CONFLICT
            return idx + bankOffset(idx);
#else
            return idx;
#endif
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

        __global__ void kernBlockScanShared(int* data, int* blockSum, int n) {
            extern __shared__ int shared[];
            extern __shared__ int last;

            int idx = threadIdx.x + 1;
            int globalIdx = blockDim.x * blockIdx.x + threadIdx.x;

            if (globalIdx > n) {
                return;
            }

            shared[offsetAddr(idx - 1)] = data[globalIdx];
            if (idx == blockDim.x) {
                last = shared[offsetAddr(blockDim.x - 1)];
            }
            __syncthreads();
#pragma unroll
            for (int stride = 1, active = blockDim.x >> 1; stride < blockDim.x; stride <<= 1, active >>= 1) {
                if (idx <= active) {
                    int idxPa = offsetAddr(idx * stride * 2 - 1);
                    int idxCh = offsetAddr(idx * stride * 2 - 1 - stride);
                    shared[idxPa] += shared[idxCh];
                }
                __syncthreads();
            }

            if (idx == 1) {
                shared[offsetAddr(blockDim.x - 1)] = 0;
            }
            __syncthreads();
#pragma unroll
            for (int stride = blockDim.x >> 1, active = 1; stride >= 1; stride >>= 1, active <<= 1) {
                if (idx <= active) {
                    int idxPa = offsetAddr(idx * stride * 2 - 1);
                    int idxCh = offsetAddr(idx * stride * 2 - 1 - stride);
                    shared[idxPa] += shared[idxCh];
                    shared[idxCh] = shared[idxPa] - shared[idxCh];
                }
                __syncthreads();
            }
            data[globalIdx] = shared[offsetAddr(idx - 1)];

            if (idx == 1) {
                blockSum[blockIdx.x] = shared[offsetAddr(blockDim.x - 1)] + last;
            }
        }

        __global__ void kernScannedBlockAdd(int* data, const int* blockSum, int n) {
            extern __shared__ int sum;
            int idx = blockDim.x * blockIdx.x + threadIdx.x;
            if (idx >= n) {
                return;
            }

            if (threadIdx.x == 0) {
                sum = blockSum[blockIdx.x];
            }
            __syncthreads();
            data[idx] += sum;
        }

        void devScanInPlace(int* devData, int size) {
            if (size != ceilPow2(size)) {
                throw std::runtime_error("devScanInPlace:: size not pow of 2");
            }

            for (int stride = 2; stride <= size; stride <<= 1) {
                int num = size / stride;
                int blockSize = Common::getDynamicBlockSizeEXT(num);
                int blockNum = ceilDiv(num, blockSize);
                kernPartialUpSweep<<<blockNum, blockSize>>>(devData, num, stride);
            }

            cudaMemset(devData + size - 1, 0, sizeof(int));
            for (int stride = size; stride >= 2; stride >>= 1) {
                int num = size / stride;
                int blockSize = Common::getDynamicBlockSizeEXT(num);
                int blockNum = ceilDiv(num, blockSize);
                kernPartialDownSweep<<<blockNum, blockSize>>>(devData, num, stride);
            }
        }

        void devBlockScanInPlaceShared(int* devData, int* devBlockSum, int size, int blockSize) {
            if (size % blockSize != 0) {
                throw std::runtime_error("devBlockScanInPlaceShared:: size not multiple of BlockSize");
            }
            kernBlockScanShared<<<size / blockSize, blockSize>>>(devData, devBlockSum, size);
        }

        void devScanInPlaceShared(int* devData, int size) {
            if (size % SharedScanBlockSize != 0 || size <= SharedScanBlockSize) {
                throw std::runtime_error("devScanInPlaceShared:: size not multiple of BlockSize");
            }

            DevSharedScanAuxBuffer<int> devBuf(size, SharedScanBlockSize);
            cudaMemcpy(devBuf.data(), devData, size * sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToDevice);

            timer().startGpuTimer();
            for (int i = 0; i + 1 < devBuf.numLayers(); i++) {
                devBlockScanInPlaceShared(devBuf[i], devBuf[i + 1], devBuf.sizeAt(i), SharedScanBlockSize);
            }

            for (int i = devBuf.numLayers() - 2; i > 0; i--) {
                devScannedBlockAdd(devBuf[i - 1], devBuf[i], devBuf.sizeAt(i - 1), SharedScanBlockSize);
            }
            timer().endGpuTimer();

            cudaMemcpy(devData, devBuf.data(), size * sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToDevice);
            devBuf.destroy();
        }

        void devScannedBlockAdd(int* devData, int* devBlockSum, int n, int blockSize) {
            if (n % blockSize != 0) {
                throw std::runtime_error("devScannedBlockAdd:: size not multiple of BlockSize");
            }
            kernScannedBlockAdd<<<n / blockSize, blockSize>>>(devData, devBlockSum, n);
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

        void scanShared(int* out, const int* in, int n, int blockSize) {
            // Just to keep the edge case correct
            // If n <= blockSize, there's no need to perform a GPU scan
            if (n <= blockSize) {
                out[0] = 0;
                for (int i = 1; i < n; i++) {
                    out[i] = out[i - 1] + in[i - 1];
                }
                return;
            }

            DevSharedScanAuxBuffer<int> devBuf(n, SharedScanBlockSize);
            cudaMemcpy(devBuf.data(), in, n * sizeof(int), cudaMemcpyKind::cudaMemcpyHostToDevice);

            timer().startGpuTimer();
            for (int i = 0; i + 1 < devBuf.numLayers(); i++) {
                devBlockScanInPlaceShared(devBuf[i], devBuf[i + 1], devBuf.sizeAt(i), blockSize);
            }

            for (int i = devBuf.numLayers() - 2; i > 0; i--) {
                devScannedBlockAdd(devBuf[i - 1], devBuf[i], devBuf.sizeAt(i - 1), blockSize);
            }
            timer().endGpuTimer();

            cudaMemcpy(out, devBuf.data(), n * sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost);
            devBuf.destroy();
        }

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param out    The array into which to store elements.
         * @param in     The array of elements to compact.
         * @param n      The number of elements in idata.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int* out, const int* in, int n)
        {
            int* devIn, * devOut;
            cudaMalloc(&devIn, n * sizeof(int));
            cudaMalloc(&devOut, n * sizeof(int));
            cudaMemcpy(devIn, in, n * sizeof(int), cudaMemcpyKind::cudaMemcpyHostToDevice);

            int size = ceilPow2(n);
            int* devIndices;
            cudaMalloc(&devIndices, size * sizeof(int));

            timer().startGpuTimer();

            int blockSize = Common::getDynamicBlockSizeEXT(n);
            int blockNum = ceilDiv(n, blockSize);

            Common::kernMapToBoolean<<<blockNum, blockSize>>>(n, devIndices, devIn);
            devScanInPlace(devIndices, size);
            Common::kernScatter<<<blockNum, blockSize>>>(n, devOut, devIn, devIn, devIndices);

            timer().endGpuTimer();

            int compactedSize;
            cudaMemcpy(&compactedSize, devIndices + n - 1, sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost);
            compactedSize += (in[n - 1] != 0);

            cudaMemcpy(out, devOut, compactedSize * sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost);

            cudaFree(devIndices);
            cudaFree(devIn);
            cudaFree(devOut);

            return compactedSize;
        }

        int compactShared(int* out, const int* in, int n)
        {
            int* devIn, * devOut;
            cudaMalloc(&devIn, n * sizeof(int));
            cudaMalloc(&devOut, n * sizeof(int));
            cudaMemcpy(devIn, in, n * sizeof(int), cudaMemcpyKind::cudaMemcpyHostToDevice);

            int size = ceilPow2(n);
            int* devIndices;
            cudaMalloc(&devIndices, size * sizeof(int));

            DevSharedScanAuxBuffer<int> devBuf(n, SharedScanBlockSize);

            timer().startGpuTimer();

            int blockSize = Common::getDynamicBlockSizeEXT(n);
            int blockNum = ceilDiv(n, blockSize);
            Common::kernMapToBoolean<<<blockNum, blockSize>>>(n, devIndices, devIn);
            cudaMemcpy(devBuf.data(), devIndices, n * sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToDevice);

            for (int i = 0; i + 1 < devBuf.numLayers(); i++) {
                devBlockScanInPlaceShared(devBuf[i], devBuf[i + 1], devBuf.sizeAt(i), SharedScanBlockSize);
            }
            for (int i = devBuf.numLayers() - 2; i > 0; i--) {
                devScannedBlockAdd(devBuf[i - 1], devBuf[i], devBuf.sizeAt(i - 1), SharedScanBlockSize);
            }

            cudaMemcpy(devIndices, devBuf.data(), n * sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToDevice);
            Common::kernScatter<<<blockNum, blockSize>>>(n, devOut, devIn, devIn, devIndices);

            timer().endGpuTimer();

            int compactedSize;
            cudaMemcpy(&compactedSize, devIndices + n - 1, sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost);
            compactedSize += (in[n - 1] != 0);

            cudaMemcpy(out, devOut, compactedSize * sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost);
            cudaFree(devIndices);
            cudaFree(devIn);
            cudaFree(devOut);
            devBuf.destroy();

            return compactedSize;
        }
    }
}
