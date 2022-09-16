#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include <thrust/remove.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include "common.h"
#include "thrust.h"

namespace StreamCompaction {
    namespace Thrust {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        struct Pred {
            __device__ bool operator()(int x) const {
                return !x;
            }
        };

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *out, const int *in) {
            thrust::device_vector<int> dev_in(in, in + n);
            thrust::device_vector<int> dev_out(n);

            timer().startGpuTimer();
            // TODO use `thrust::exclusive_scan`
            // example: for device_vectors dv_in and dv_out:

            thrust::exclusive_scan(dev_in.begin(), dev_in.end(), dev_out.begin());
            timer().endGpuTimer();
            
            thrust::copy(dev_out.begin(), dev_out.end(), out);
        }

        int compact(int n, int* out, const int* in) {
            thrust::device_vector<int> dev_in(in, in + n);
            auto it = thrust::remove_if(dev_in.begin(), dev_in.end(), Pred());

            int ret = thrust::distance(dev_in.begin(), it);
            thrust::copy(dev_in.begin(), it, out);
            return ret;
        }

        void sort(int n, int* out, const int* in) {
            thrust::device_vector<int> dev_in(in, in + n);
            thrust::sort(dev_in.begin(), dev_in.end());
            thrust::copy(dev_in.begin(), dev_in.end(), out);
        }
    }
}
