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
            out[!(val & filter) ? pos : numZero + idx - pos] = val;
        }

        void sort(int* out, const int* in, int n) {
            int* data, * buf;
            cudaMalloc(&data, n * sizeof(int));
            cudaMalloc(&buf, n * sizeof(int));
            cudaMemcpy(data, in, n * sizeof(int), cudaMemcpyKind::cudaMemcpyHostToDevice);

            int* indices;
            int size = ceilPow2(n);
            cudaMalloc(&indices, size * sizeof(int));

            timer().startGpuTimer();

            for (uint32_t bit = 1; bit < 0x80000000u; bit <<= 1) {
                int blockSize = Common::getDynamicBlockSizeEXT(n);
                int blockNum = (n + blockSize - 1) / blockSize;

                kernMapToBool<<<blockNum, blockSize>>>(indices, data, n, bit);
                Efficient::devScanInPlace(indices, size);

                int numZero;
                cudaMemcpy(&numZero, indices + n - 1, sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost);
                int dataN;
                cudaMemcpy(&dataN, data + n - 1, sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost);
                numZero += (dataN & bit) == 0;

                kernScatter<<<blockNum, blockSize>>>(buf, data, indices, n, bit, numZero);
                std::swap(data, buf);
            }
            cudaMemcpy(out, data, n * sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost);

            cudaFree(data);
            cudaFree(buf);
            cudaFree(indices);

            timer().endGpuTimer();
        }
    }
}