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
            int* dev_out;
            int* dev_in;
            cudaMalloc((void**)&dev_out, n * sizeof(int));
            cudaMalloc((void**)&dev_in, n * sizeof(int));
            cudaMemcpy(dev_out, odata, sizeof(int) * n, cudaMemcpyHostToDevice);
            cudaMemcpy(dev_in, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
            thrust::device_ptr<int> dev_thrust_out(dev_out);
            thrust::device_ptr<int> dev_thrust_in(dev_in);
            timer().startGpuTimer();
            // TODO use `thrust::exclusive_scan`
            // example: for device_vectors dv_in and dv_out:
            // thrust::exclusive_scan(dv_in.begin(), dv_in.end(), dv_out.begin());
            thrust::exclusive_scan(dev_thrust_in, dev_thrust_in + n, dev_thrust_out);
            timer().endGpuTimer();
            cudaMemcpy(odata, dev_out, sizeof(int) * n, cudaMemcpyDeviceToHost);
            cudaFree(dev_out);
            cudaFree(dev_in);
        }
    }
}
