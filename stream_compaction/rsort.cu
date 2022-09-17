#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "rsort.h"
#include "efficient.h"

namespace StreamCompaction {
    namespace RadixSort {
        using StreamCompaction::Common::PerformanceTimer;

        PerformanceTimer& timer() {
            static PerformanceTimer timer;
            return timer;
        }

        /** computes the boolean array for radix sort pass d
        */
        __global__ void kernComputeBoolean(int const N, int const d, int const* in, int* out_bool) {
            int self = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (self >= N) {
                return;
            }
            out_bool[self] = (in[self] >> d & 1) ? 1 : 0;
        }

        /** computes auxilary arrays
        */
        __global__ void kernInvertBoolean(int const N, int const* in, int* out) {
            int self = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (self >= N) {
                return;
            }
            out[self] = 1 - in[self];
        }
        __global__ void kernComputeIndices(int const N, int total_falses, int const* in_f, int const* in_b, int* out) {
            int self = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (self >= N) {
                return;
            }
            out[self] = in_b[self] ? self - in_f[self] + total_falses : in_f[self];
        }
        __global__ void kernShuffle(int const N, int const* in, int const* indices, int* out) {
            int self = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (self >= N) {
                return;
            }
            out[indices[self]] = in[self];
        }
        void sort(int n, int* out, int const* in) {
            int dlim = 0;
            for (int i = 0; i < n; ++i) {
                int bit_count = 1;
                for (int j = in[i]; j; j >>= 1) {
                    ++bit_count;
                }
                dlim = std::max(dlim, bit_count);
            }

            dim3 nblocks((n + blockSize - 1) / blockSize);

            int* dev_in;
            int* dev_out;
            int* dev_bool;
            int* dev_e;
            int* dev_f = nullptr;
            int* dev_d;
            int pow2len;

            ALLOC(dev_in, n);
            H2D(dev_in, in, n);

            ALLOC(dev_out, n);
            if (n == 1) D2D(dev_out, dev_in, n);
            
            ALLOC(dev_bool, n);
            ALLOC(dev_e, n);
            ALLOC(dev_d, n);

            timer().startGpuTimer();
            for (int d = 0; d < dlim; ++d) {
                kernComputeBoolean KERN_PARAM(nblocks, blockSize) (n, d, dev_in, dev_bool);
                kernInvertBoolean  KERN_PARAM(nblocks, blockSize) (n, dev_bool, dev_e);
                
                pow2len = Common::makePowerTwoLength(n, dev_e, &dev_f, Common::MakePowerTwoLengthMode::DeviceToDevice);
                Efficient::scan_impl(pow2len, dev_f);
                
                int total_falses = getGPU(dev_f, n-1) + getGPU(dev_e, n-1);
                kernComputeIndices KERN_PARAM(nblocks, blockSize) (n, total_falses, dev_f, dev_bool, dev_d);
                kernShuffle        KERN_PARAM(nblocks, blockSize) (n, dev_in, dev_d, dev_out);
                
                std::swap(dev_in, dev_out);

                //PRINT_GPU(dev_bool, n);
                //PRINT_GPU(dev_e, n);
                //PRINT_GPU(dev_f, n);
                //PRINT_GPU(dev_d, n);
                //PRINT_GPU(dev_in, n);
                //PRINT_GPU(dev_out, n);
            }
            timer().endGpuTimer();

            // ensure out is the result
            if (dlim & 1) {
                std::swap(dev_in, dev_out);
            }

            //PRINT_GPU(dev_in, n);
            //PRINT_GPU(dev_out, n);

            D2H(out, dev_out, n);

            FREE(dev_out);
            FREE(dev_bool);
            FREE(dev_e);
            if(dev_f)
                FREE(dev_f);
            FREE(dev_d);
            FREE(dev_in);
        }
    }
}