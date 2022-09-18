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
            thrust::device_vector<int> thrust_dv_idata(idata, idata + n);
            thrust::device_vector<int> thrust_dv_odata(odata, odata + n);

            timer().startGpuTimer();

            thrust::exclusive_scan(thrust_dv_idata.begin(), thrust_dv_idata.end(), thrust_dv_odata.begin());

            timer().endGpuTimer();

            cudaMemcpy(odata, (thrust_dv_odata.data()).get(), sizeof(int) * n, cudaMemcpyDeviceToHost);
        }
    }
}
