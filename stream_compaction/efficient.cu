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
#define SHARED_OPT_BLOCK_SIZE 1024
        __global__ void kernSharedScan(int N, int* out, int const* in) {
            __shared__ int temp[2 * SHARED_OPT_BLOCK_SIZE];
            int thid = threadIdx.x;
            int offset = 1;

            // up sweep
            temp[2 * thid] = in[2 * thid];
            temp[2 * thid + 1] = in[2 * thid + 1];
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
        }


        /** 
        * performs in-place exclusive scan on a GPU buffer
        * n must be a power of two
        */
        inline void scan_impl(int n, int* dev_in_out) {
#ifdef SHARED_OPT
            if (n < 1024) {
                kernSharedScan KERN_PARAM(1, SHARED_OPT_BLOCK_SIZE) (n, dev_in_out, dev_in_out);
            } else {
                assert(false);
                //int nblocks = (n + SHARED_THREAD_NUM - 1) / SHARED_THREAD_NUM;
                //kernSharedScan KERN_PARAM(nblocks, SHARED_THREAD_NUM) 
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
            int ret = getGPU(dev_indices, pow2len, pow2len - 1);

            D2H(out, dev_out, ret);
            FREE(dev_out);
            FREE(dev_in);
            FREE(dev_indices);
            FREE(dev_bool);
            
            return ret;
        }
    }
}
