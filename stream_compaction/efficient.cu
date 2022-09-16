#include <cuda.h>
#include <cuda_runtime.h>

#include "common.h"
#include "efficient.h"
#include <assert.h>

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        /**
        * Implementation of scan using global memory
        */
        __global__ void kernUpSweep(int N, int d, int* out) {
            int self = (blockIdx.x * blockDim.x) + threadIdx.x;
            int p0 = 1 << d;
            int p1 = 1 << (d + 1);
            if (self >= N || self % p1) {
                return;
            }
            out[self + p1 - 1] += out[self + p0 - 1];
        }
        __global__ void kernDownSweep(int N, int d, int* out, const int dlim) {
            int self = (blockIdx.x * blockDim.x) + threadIdx.x;
            int p0 = 1 << d;
            int p1 = 1 << (d + 1);
            if (self >= N || self % p1) {
                return;
            }
            int left = self + p0 - 1;
            int cur = self + p1 - 1;
            int save = out[left];
            out[left] = out[cur];
            out[cur] += save;
        }

        /**
        *  Implementation of scan using shared memory
        *  reference: GPU Gem3 Listing 39-2
        */
#define SHARED_OPT
#define SHARED_OPT_BSIZE 2 // how many threads per block in shared memory implementation
#define SHARED_OPT_MAX_ARR_SIZE (2 * SHARED_OPT_BSIZE) // maximum array that can be processed by a block in shared memory implementation
        __global__ void kernSharedScan(int N, int* out, int const* in, bool inclusive) {
            __shared__ int temp[SHARED_OPT_MAX_ARR_SIZE];
            int blkid = blockIdx.x;
            int thid = threadIdx.x;
            int offset = 1;

            // pad input and output so we r/w to the correct chunk
            out += blkid * SHARED_OPT_MAX_ARR_SIZE;
            in += blkid * SHARED_OPT_MAX_ARR_SIZE;

            // up sweep
            int saves[2];
            saves[0] = temp[2 * thid] = in[2 * thid];
            saves[1] = temp[2 * thid + 1] = in[2 * thid + 1];

            for (int d = N >> 1; d > 0; d >>= 1) {
                __syncthreads();
                if (thid < d) {
                    int ai = offset * (2 * thid + 1) - 1;
                    int bi = offset * (2 * thid + 2) - 1;
                    temp[bi] += temp[ai];
                }
                offset <<= 1;
            }

            // downsweep
            if (!thid) {
                temp[N - 1] = 0;
            }

            for (int d = 1; d < N; d <<= 1) {
                offset >>= 1;
                __syncthreads();
                if (thid < d) {
                    int ai = offset * (2 * thid + 1) - 1;
                    int bi = offset * (2 * thid + 2) - 1;
                    int save = temp[ai];
                    temp[ai] = temp[bi];
                    temp[bi] += save;
                }
            }
            __syncthreads();

            out[2 * thid] = temp[2 * thid];
            out[2 * thid + 1] = temp[2 * thid + 1];
            if (inclusive) {
                out[2 * thid] += saves[0];
                out[2 * thid + 1] += saves[1];
            }
        }

        __global__ void kernOffset(int N, int* in, int* out, bool plus) {
            int self = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (self >= N) {
                return;
            }
            if (plus) {
                out[self] += in[self];
            } else {
                out[self] -= in[self];
            }
        }
        /**
        *  Compute block increments if the array was split into several blocks
        */
        __global__ void kernComputeStride(int N, int* in, int* out) {
            int self = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (self >= N) {
                return;
            }
            out[self] = (in + self * SHARED_OPT_MAX_ARR_SIZE)[SHARED_OPT_MAX_ARR_SIZE -1];
        }
        /**
        *  Compute the final prefix sum based on block increments
        */
        __global__ void kernComputeFinalArray(int N, int* strides, int* out) {
            int self = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (self >= N) {
                return;
            }
            out[self] += strides[self / SHARED_OPT_MAX_ARR_SIZE];
        }

        /** 
        * performs in-place exclusive scan on a GPU buffer
        * n must be a power of two
        */
        void scan_impl(int n, int* dev_in_out, bool inclusive) {
            assert((n & -n) == n);
            static_assert((SHARED_OPT_MAX_ARR_SIZE & -SHARED_OPT_MAX_ARR_SIZE) == SHARED_OPT_MAX_ARR_SIZE, "max array size must be pow of 2");
#ifdef SHARED_OPT
            if (n <= SHARED_OPT_MAX_ARR_SIZE) {
                kernSharedScan KERN_PARAM(1, SHARED_OPT_BSIZE) (n, dev_in_out, dev_in_out, inclusive);
                cudaDeviceSynchronize();
            } else {
                assert(n % SHARED_OPT_MAX_ARR_SIZE == 0);

                // split into chunks if array cannot fit into shared memory
                int nblocks = n / SHARED_OPT_MAX_ARR_SIZE;

                // note: use SHARED_OPT_BSIZE because each thread is responsible for 2 values in the array
                kernSharedScan KERN_PARAM(nblocks, SHARED_OPT_BSIZE) (SHARED_OPT_MAX_ARR_SIZE, dev_in_out, dev_in_out, true);
                cudaDeviceSynchronize();

                int* stride;
                ALLOC(stride, nblocks);

                // the dimensions here are arbitrary, as long as it covers the stride array
                // which has a size of N / SHARED_OPT_MAX_ARR_SIZE
                int nblocks2 = (nblocks + SHARED_OPT_BSIZE - 1) / SHARED_OPT_BSIZE;
                kernComputeStride KERN_PARAM(nblocks2, SHARED_OPT_BSIZE) (nblocks, dev_in_out, stride);
                cudaDeviceSynchronize();

                // PRINT_GPU(dev_in_out, n);
                scan_impl(nblocks, stride, false);
                // PRINT_GPU(stride, nblocks);

                kernComputeFinalArray KERN_PARAM(nblocks, SHARED_OPT_MAX_ARR_SIZE) (n, stride, dev_in_out);
                // PRINT_GPU(dev_in_out, n);
                cudaDeviceSynchronize();

                if (!inclusive) {
                    // inclusive to exclusive
                    int* tmp;
                    ALLOC(tmp, n);
                    D2D(tmp+1, dev_in_out, n-1);
                    setGPU(tmp, 0, 0);
                    D2D(dev_in_out, tmp, n);
                    FREE(tmp);
                }

                FREE(stride);
            }
#else
            dim3 nblocks((n + blockSize - 1) / blockSize);
            int dlim = ilog2ceil(n);
            for (int d = 0; d < dlim; ++d) {
                kernUpSweep KERN_PARAM(nblocks, blockSize) (n, d, dev_in_out);
            }
            // set root to zero
            int zero = 0;
            H2D(dev_in_out + n - 1, &zero, 1);

            for (int d = dlim - 1; d >= 0; --d) {
                kernDownSweep KERN_PARAM(nblocks, blockSize) (n, d, dev_in_out, dlim);
            }
#endif // SHARED_OPT

        }
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *out, const int *in) {
            // TODO
            int old_len = n;
            int* dev_out = nullptr;

            n = Common::makePowerTwoLength(n, in, &dev_out, Common::MakePowerTwoLengthMode::HostToDevice);
            
            timer().startGpuTimer();
            scan_impl(n, dev_out);
            timer().endGpuTimer();

            D2H(out, dev_out, old_len);
            FREE(dev_out);
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
        int compact(int n, int *out, const int *in) {
            dim3 nblocks((n + blockSize - 1) / blockSize);
            int dlim = ilog2ceil(n);
            int* dev_bool;
            int* dev_indices = nullptr;
            int* dev_in;
            int* dev_out;
            ALLOC(dev_bool, n); // filled by mapToBoolean
            ALLOC(dev_in, n); H2D(dev_in, in, n);
            ALLOC(dev_out, n);  // filled by scatter
            int pow2len;

            timer().startGpuTimer();
            {
                // TODO
                Common::kernMapToBoolean KERN_PARAM(nblocks, blockSize) (n, dev_bool, dev_in);

                // pad input if not power of 2
                pow2len = Common::makePowerTwoLength(n, dev_bool, &dev_indices, Common::MakePowerTwoLengthMode::DeviceToDevice);
                scan_impl(pow2len, dev_indices);

                Common::kernScatter KERN_PARAM(nblocks, blockSize) (n, dev_out, dev_in, dev_bool, dev_indices);
            }
            timer().endGpuTimer();
            int ret = getGPU(dev_indices, pow2len - 1);

            D2H(out, dev_out, ret);
            FREE(dev_out);
            FREE(dev_in);
            FREE(dev_indices);
            FREE(dev_bool);
            
            return ret;
        }
    }
}
