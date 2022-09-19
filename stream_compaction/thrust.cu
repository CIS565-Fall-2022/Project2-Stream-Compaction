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

        int* dev_idata;
        int* dev_odata;
        int* dev_buf;

        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void toInclusive(int N, int* idata, int* odata, int* buf) {
            int k = threadIdx.x + (blockIdx.x * blockDim.x);
            if (k >= N) {
                return;
            }
            if (k < N - 1) {
                odata[k] = buf[k + 1];
            }
            else {
                odata[k] = buf[k] + idata[k];
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
            // TODO use `thrust::exclusive_scan`
            // example: for device_vectors dv_in and dv_out:
            // thrust::exclusive_scan(dv_in.begin(), dv_in.end(), dv_out.begin());
            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            cudaMalloc((void**)&dev_odata, n * sizeof(int));
            cudaMalloc((void**)&dev_buf, n * sizeof(int));

            cudaMemcpy(dev_idata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);


            thrust::device_ptr<int> dev_thrust_idata = thrust::device_ptr<int>(dev_idata);
            thrust::device_ptr<int> dev_thrust_odata = thrust::device_ptr<int>(dev_odata);

            thrust::inclusive_scan(dev_thrust_idata, dev_thrust_idata+n, dev_thrust_odata);
            //cudaMemcpy(dev_buf, dev_odata, sizeof(int) * arrLen, cudaMemcpyDeviceToDevice);

            cudaMemcpy(odata, dev_odata, sizeof(int) * n, cudaMemcpyDeviceToHost);

            timer().endGpuTimer();
        }
    }
}
