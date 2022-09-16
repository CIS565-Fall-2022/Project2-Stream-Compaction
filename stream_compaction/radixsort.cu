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

        void sort(int* out, const int* in, int n, ScanMethod method)
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

                if (method == ScanMethod::Shared) {
                    Efficient::devScanInPlaceShared(devIndices, size);
                }
                else {
                    Efficient::devScanInPlace(devIndices, size);
                }

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
    }
}