#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
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
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            thrust::device_vector<int> thrustIn(idata, idata + n);
            thrust::device_vector<int> thrustOut(n);
            timer().startGpuTimer();
            thrust::exclusive_scan(thrustIn.begin(), thrustIn.end(), thrustOut.begin());
            timer().endGpuTimer();
            thrust::copy(thrustOut.begin(), thrustOut.end(), odata);
        }
    }
}
