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
        * performs a scan at iteration d with offset p = 1 << d, reads from in and writes to out
        */
        __global__ void kernNaiveScan(int N, int p, int *in, int *out) {
            int self = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (self >= N) {
                return;
            }

            // int p = 1 << d;
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

            int* dev_in = nullptr;
            int* dev_out;

            int pow2len = Common::makePowerTwoLength(n, in, &dev_in, Common::MakePowerTwoLengthMode::HostToDevice);
            dim3 nblocks((pow2len + blockSize - 1) / blockSize);

            // ALLOC(dev_in, n);
            ALLOC(dev_out, pow2len);
            // H2D(dev_in, in, n);
            if (n <= 1) {
                H2D(dev_out, in, n);
            }

            // I commented out some code because apparently they cause this function to fail when input size >= 2^27
            // No idea why to this date

            timer().startGpuTimer();
            for (int d = 0; d < dlim; ++d) {
                kernNaiveScan KERN_PARAM(nblocks, blockSize) (pow2len, 1 << d, dev_in, dev_out);
                // cudaDeviceSynchronize();
                D2D(dev_in, dev_out, pow2len);
                // std::swap(dev_in, dev_out);
            }
            timer().endGpuTimer();

            // ensure out is the result
            //if (dlim & 1) {
            //    std::swap(dev_in, dev_out);
            //}
            // inclusive to exclusive
            D2H(out+1, dev_out, n-1);
            out[0] = 0;
            FREE(dev_in);
            FREE(dev_out);
        }
    }
}