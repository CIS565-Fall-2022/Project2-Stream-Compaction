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
            if (d == dlim - 1 && self == N - 1) {
                out[self] = 0;
                return;
            } else if (self >= N || self % p1) {
                return;
            }
            int left = self + p0 - 1;
            int cur = self + p1 - 1;
            int save = out[left];
            out[left] = out[cur];
            out[cur] += save;
        }

        static inline void scan_impl(int n, int dlim, int* dev_in_out) {
            dim3 nblocks((n + blockSize - 1) / blockSize);

            for (int d = 0; d < dlim; ++d) {
                kernUpSweep KERN_PARAM(nblocks, blockSize) (n, d, dev_in_out);
            }
            for (int d = dlim - 1; d >= 0; --d) {
                kernDownSweep KERN_PARAM(nblocks, blockSize) (n, d, dev_in_out, dlim);
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *out, const int *in) {
            // TODO
            int old_len = n;
            int dlim = ilog2ceil(n);

            if (n == 1 || (n & -n != n)) {
                // n is not a power of two
                n = 1 << dlim;
            }

            int* dev_out;
            ALLOC(dev_out, n);
            H2D(dev_out, in, old_len);
            if (n != old_len) {
                MEMSET(dev_out + old_len, 0, (n - old_len) * sizeof(int));
            }

            timer().startGpuTimer();
            scan_impl(n, dlim, dev_out);
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
            int old_len = n;
            int dlim = ilog2ceil(n);

            if (n == 1 || (n & -n != n)) {
                // n is not a power of two
                n = 1 << dlim;
            }


            int* dev_bool;
            int* dev_indices;
            int* dev_in;
            int* dev_out;
            ALLOC(dev_bool, n);
            ALLOC(dev_indices, n);
            ALLOC(dev_in, n);
            ALLOC(dev_out, n);

            // pad input if not power of 2
            H2D(dev_in, in, old_len);
            if (n != old_len) {
                MEMSET(dev_in + old_len, 0, (n - old_len) * sizeof(int));
            }

            timer().startGpuTimer();
            // TODO
            Common::kernMapToBoolean KERN_PARAM(nblocks, blockSize) (n, dev_bool, dev_in);
            D2D(dev_indices, dev_bool, n);
            scan_impl(n, dlim, dev_indices);
            Common::kernScatter KERN_PARAM(nblocks, blockSize) (n, dev_out, dev_in, dev_bool, dev_indices);

            timer().endGpuTimer();

            int pret[1];
            D2H(pret, dev_indices + n - 1, 1);
            D2H(out, dev_out, *pret);
            FREE(dev_out);
            FREE(dev_in);
            FREE(dev_indices);
            FREE(dev_bool);
            
            return *pret;
        }
    }
}
