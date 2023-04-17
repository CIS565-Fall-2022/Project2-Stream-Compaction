#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

const int blockSize = 128;

__device__ inline int twoPow(int d) {
    return (1 << (d));
}

inline int twoPowHost(int d) {
    return (1 << (d));
}

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernUpSweep(int n, int d, int *x) {
            int idx = blockDim.x * blockIdx.x + threadIdx.x;
            if (idx >= n) return;
            if (idx % twoPow(d + 1) == 0)
                x[idx + twoPow(d + 1) - 1] += x[idx + twoPow(d) - 1];
        }

        __global__ void kernDownSweep(int n, int d, int *x) {
            int idx = blockDim.x * blockIdx.x + threadIdx.x;
            if (idx >= n) return;
            if (idx % twoPow(d + 1) == 0) {
                int tmp = x[idx + twoPow(d) - 1];
                x[idx + twoPow(d) - 1] = x[idx + twoPow(d + 1) - 1];
                x[idx + twoPow(d + 1) - 1] += tmp;
            }
        }

        /**
         * 在这个实现中，对于输入数组的长度不是2的幂次的情况，会将其扩展到最小的2的幂次大小，
         * 这样做的好处是可以将输入数组分成规模相同的子数组，便于并行计算。
         * 在计算完前缀和后，把多余的部分（即最后的3个元素）置为0即可。
         *
         * 具体来说，在UpSweep阶段，每个线程处理一个数组元素，
         * 如果这个元素的下标满足idx % 2^(d+1) == 0，
         * 则将这个元素的值加上它前面距离它2^d个元素的元素的值。
         * 这样就将每个距离为2^d的元素对应的和计算出来。
         * 这个过程一共执行log2(size)次，每一次处理的距离都是上一次的两倍。
         *
         * 在DownSweep阶段，先将最后一个元素置为0，然后从最后一层开始，
         * 每个线程处理一个数组元素，如果这个元素的下标满足idx % 2^(d+1) == 0，
         * 则将这个元素的值和它前面距离它2^d个元素的元素的值交换，并将前面的值加到后面的值上。
         * 这样就将每个距离为2^d的元素对应的和从下往上传递。
         * 同样，这个过程也是执行log2(size)次。最后，整个数组的前缀和就计算完成了。
         */

        /**
         * In this implementation, if the length of the input array is not a power of 2,
         * it will be extended to the smallest power of 2 size.
         * This is done to facilitate parallel computation by dividing the input array into equally-sized subarrays.
         * After computing the prefix sum, the excess part of the array (i.e., the last 3 elements) is set to 0.
         *
         * Specifically, in the UpSweep phase, each thread processes one element of the array.
         * If the index of this element satisfies idx % 2^(d+1) == 0,
         * then the value of this element is added to the value of the element located 2^d positions in front of it.
         * This way, the sums of every two elements that are 2^d apart are calculated. This process is repeated log2(size) times,
         * where each iteration processes elements that are twice as far apart as the previous iteration.
         *
         * In the DownSweep phase, the last element of the array is set to 0.
         * Starting from the last level, each thread processes one element of the array.
         * If the index of this element satisfies idx % 2^(d+1) == 0,
         * then the value of this element is swapped with the value of the element located 2^d positions in front of it,
         * and the value of the latter element is added to the former element.
         * This way, the sums of every two elements that are 2^d apart are propagated upwards from the bottom of the array.
         * Again, this process is repeated log2(size) times.
         * Finally, the prefix sum of the entire array is computed.
         */

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            int size = twoPowHost(ilog2ceil(n)); // ensure the size is pow of 2
            // for example:
            // if n = 253, let size equal to 256.
            // ilog2ceil(253) = [log2(253)] + 1 = log2(128) + 1 = 8
            // twoPowHost(8) = 256
            dim3 blockPerGrids((size + blockSize - 1) / blockSize);
            int *dev_idata;

            cudaMalloc((void **)&dev_idata, size * sizeof(int));

            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            timer().startGpuTimer();
            // TODO
            // UpSweep
            for (int d = 0; d < ilog2ceil(size); d++) {
                kernUpSweep<<<blockPerGrids, blockSize>>>(n, d, dev_idata);
                cudaDeviceSynchronize(); // ensure that the previous cuda jobs have completed
            }
            // set the last value of dev_idata to zero
            cudaMemset(dev_idata + size - 1, 0, sizeof(int));

            // DownSweep
            for (int d = ilog2ceil(size) - 1; d >= 0; d--) {
                kernDownSweep<<<blockPerGrids, blockSize>>>(n, d, dev_idata);
                cudaDeviceSynchronize();
            }
            timer().endGpuTimer();
            cudaMemcpy(odata, dev_idata, n * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(dev_idata);
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
            int *dev_bools;
            int *dev_indices;
            int *dev_idata;
            int *dev_odata;
            int size = twoPowHost(ilog2ceil(n));
            int cnt = 0;

            dim3 blockPerGrids((n + blockSize - 1) / blockSize);
            dim3 fullBlockPerGrids((size + blockSize - 1) / blockSize);

            cudaMalloc((void **)&dev_bools, size * sizeof(int));
            cudaMalloc((void **)&dev_indices, size * sizeof(int));
            cudaMalloc((void **)&dev_idata, size * sizeof(int));
            cudaMalloc((void **)&dev_odata, size * sizeof(int));

            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            timer().startGpuTimer();
            // TODO
            Common::kernMapToBoolean<<<blockPerGrids, blockSize>>>(n, dev_bools, dev_idata);
            cudaDeviceSynchronize();
            cudaMemcpy(dev_indices, dev_bools, n * sizeof(int), cudaMemcpyDeviceToDevice);

            // scan
            for (int d = 0; d < ilog2ceil(size); d++) {
                kernUpSweep<<<fullBlockPerGrids, blockSize>>>(n, d, dev_indices);
                cudaDeviceSynchronize();
            }

            cudaMemset(dev_indices + size - 1, 0, sizeof(int));

            for (int d = ilog2ceil(size) - 1; d >= 0; d--) {
                kernDownSweep<<<fullBlockPerGrids, blockSize>>>(n, d, dev_indices);
                cudaDeviceSynchronize();
            }

            Common::kernScatter<<<blockPerGrids, blockSize>>>(n, dev_odata, dev_idata, dev_bools, dev_indices);
            timer().endGpuTimer();

            cudaMemcpy(&cnt, dev_indices + size - 1, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(odata, dev_odata, cnt * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(dev_idata);
            cudaFree(dev_odata);
            cudaFree(dev_indices);
            cudaFree(dev_bools);

            return cnt;
        }
    }
}
