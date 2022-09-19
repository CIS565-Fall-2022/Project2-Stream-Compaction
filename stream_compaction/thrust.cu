#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include "common.h"
#include "thrust.h"
#include <thrust/execution_policy.h>

namespace StreamCompaction {
    namespace Thrust {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            // TODO use `thrust::exclusive_scan`
            // example: for device_vectors dv_in and dv_out:
            // thrust::exclusive_scan(dv_in.begin(), dv_in.end(), dv_out.begin());

            thrust::host_vector<int> host_in(idata, idata + n);
            thrust::device_vector<int> dev_in = host_in;
            thrust::device_vector<int> dev_out(n);


            timer().startGpuTimer();
            thrust::exclusive_scan(dev_in.begin(), dev_in.end(), dev_out.begin());
            timer().endGpuTimer();
            thrust::copy(dev_out.begin(), dev_out.end(), odata);

        }
    }
}
