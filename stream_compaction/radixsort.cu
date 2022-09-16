#include "radixsort.h"
#include "efficient.h"

namespace StreamCompaction {
    namespace RadixSort {
        StreamCompaction::Common::PerformanceTimer& timer() {
            static StreamCompaction::Common::PerformanceTimer timer;
            return timer;
        }

        __global__ void kernMapToBool(int* out, const int* in, int n, int filter) {
            int idx = blockDim.x * blockIdx.x + threadIdx.x;
            if (idx >= n) {
                return;
            }
            out[idx] = (in[idx] & filter) == 0;
        }

        __global__ void kernScatter(int* out, const int* data, const int* indices, int n, int filter, int numZero) {
            int idx = blockDim.x * blockIdx.x + threadIdx.x;
            if (idx >= n) {
                return;
            }
            int val = data[idx];
            int pos = indices[idx];
            out[(val & filter) ? numZero + idx - pos : pos] = val;
        }

        void sort(int* out, const int* in, int n)
        {
            int* devData, * devBuf;
            cudaMalloc(&devData, n * sizeof(int));
            cudaMalloc(&devBuf, n * sizeof(int));
            cudaMemcpy(devData, in, n * sizeof(int), cudaMemcpyKind::cudaMemcpyHostToDevice);

            int* devIndices;
            int size = ceilPow2(n);
            cudaMalloc(&devIndices, size * sizeof(int));

            timer().startGpuTimer();

            for (uint32_t bit = 1; bit < 0x80000000u; bit <<= 1) {
                int blockSize = Common::getDynamicBlockSizeEXT(n);
                int blockNum = ceilDiv(n, blockSize);

                kernMapToBool<<<blockNum, blockSize>>>(devIndices, devData, n, bit);
                Efficient::devScanInPlace(devIndices, size);

                int numZero;
                cudaMemcpy(&numZero, devIndices + n - 1, sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost);
                int dataN;
                cudaMemcpy(&dataN, devData + n - 1, sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost);
                numZero += (dataN & bit) == 0;

                kernScatter<<<blockNum, blockSize>>>(devBuf, devData, devIndices, n, bit, numZero);
                std::swap(devData, devBuf);
            }
            timer().endGpuTimer();

            cudaMemcpy(out, devData, n * sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost);
            cudaFree(devData);
            cudaFree(devBuf);
            cudaFree(devIndices);
        }

        void sortShared(int* out, const int* in, int n)
        {
            const int ScanBlockSize = 128;
            int* devData, * devBuf;
            cudaMalloc(&devData, n * sizeof(int));
            cudaMalloc(&devBuf, n * sizeof(int));
            cudaMemcpy(devData, in, n * sizeof(int), cudaMemcpyKind::cudaMemcpyHostToDevice);

            int* devIndices;
            int size = ceilPow2(n);
            cudaMalloc(&devIndices, size * sizeof(int));

            std::vector<DevMemRec<int>> sums;
            for (int i = n; i >= 1; i = ceilDiv(i, ScanBlockSize)) {
                int sz = ceilDiv(i, ScanBlockSize) * ScanBlockSize;
                int* sum;
                cudaMalloc(&sum, sz * sizeof(int));
                sums.push_back({ sum, sz, i });

                if (i == 1) {
                    break;
                }
            }
            timer().startGpuTimer();

            for (uint32_t bit = 1; bit < 0x80000000u; bit <<= 1) {
                int blockSize = Common::getDynamicBlockSizeEXT(n);
                int blockNum = ceilDiv(n, blockSize);

                kernMapToBool<<<blockNum, blockSize>>>(devIndices, devData, n, bit);

                cudaMemcpy(sums[0].ptr, devIndices, n * sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToDevice);

                for (int i = 0; i + 1 < sums.size(); i++) {
                    Efficient::devBlockScanInPlaceShared(sums[i].ptr, sums[i + 1].ptr, sums[i].size, ScanBlockSize);
                }
                for (int i = sums.size() - 2; i > 0; i--) {
                    Efficient::devScannedBlockAdd(sums[i - 1].ptr, sums[i].ptr, sums[i - 1].size, ScanBlockSize);
                }
                cudaMemcpy(devIndices, sums[0].ptr, n * sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToDevice);

                int numZero;
                cudaMemcpy(&numZero, devIndices + n - 1, sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost);
                int dataN;
                cudaMemcpy(&dataN, devData + n - 1, sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost);
                numZero += (dataN & bit) == 0;

                kernScatter<<<blockNum, blockSize>>>(devBuf, devData, devIndices, n, bit, numZero);
                std::swap(devData, devBuf);
            }
            timer().endGpuTimer();

            cudaMemcpy(out, devData, n * sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost);
            cudaFree(devData);
            cudaFree(devBuf);
            cudaFree(devIndices);

            for (auto& sum : sums) {
                cudaFree(sum.ptr);
            }
        }
    }
}