#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        // TODO: __global__
        /*
        * performs a scan at iteration d, reads from in and writes to out
        */
        __global__ void kernNaiveScan(int N, int d, int *in, int *out) {
            int self = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (self >= N) {
                return;
            }

            int p = 1 << d;
            out[self] = in[self];
            if (self >= p) {
                out[self] += in[self - p];
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *out, const int *in) {
            // TODO
            int dlim = ilog2ceil(n);
            dim3 nblocks((n + blockSize - 1) / blockSize);

            int* dev_in, * dev_out;
            ALLOC(dev_in, n);
            ALLOC(dev_out, n);
            H2D(dev_in, in, n);
            
            if (n <= 1) {
                H2D(dev_out, in, n);
            }

            timer().startGpuTimer();
            bool flip = false;
            for (int d = 0; d < dlim; ++d) {
                kernNaiveScan KERN_PARAM(nblocks, blockSize) (n, d, dev_in, dev_out);
                std::swap(dev_in, dev_out);
                flip = !flip;
            }
            timer().endGpuTimer();

            // ensure out is the result
            if (!flip) {
                std::swap(dev_in, dev_out);
            }
            // inclusive to exclusive
            D2H(out+1, dev_out, n-1);
            out[0] = 0;
            FREE(dev_in);
            FREE(dev_out);
        }
    }
}