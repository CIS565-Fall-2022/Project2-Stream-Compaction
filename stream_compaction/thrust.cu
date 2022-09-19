#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include "common.h"
#include "thrust.h"

#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)

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
            // timer().startGpuTimer();
            // TODO use `thrust::exclusive_scan`
            // example: for device_vectors dv_in and dv_out:
            // thrust::exclusive_scan(dv_in.begin(), dv_in.end(), dv_out.begin());
            int* dev_idata;
            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            checkCUDAErrorWithLine("cudaMalloc dev_idata failed!");
            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAErrorWithLine("cudaMemcpy dev_idata failed!");

            int* dev_odata;
            cudaMalloc((void**)&dev_odata, n * sizeof(int));
            checkCUDAErrorWithLine("cudaMalloc dev_odata failed!");

            thrust::device_ptr<int> dev_thrust_in = thrust::device_ptr<int>(dev_idata);
            thrust::device_ptr<int> dev_thrust_out = thrust::device_ptr<int>(dev_odata);

            // only time thrust scan
            timer().startGpuTimer();
            thrust::exclusive_scan(dev_thrust_in, dev_thrust_in + n, dev_thrust_out);
            timer().endGpuTimer();


            cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAErrorWithLine("cudaMempcy dev_odata failed!");
        }

        void sort(int n, int* odata, int* idata) {
            timer().startGpuTimer();

            int* dev_idata;
            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            checkCUDAErrorWithLine("cudaMalloc dev_idata failed!");
            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAErrorWithLine("cudaMemcpy dev_idata failed!");

            thrust::device_ptr<int> dev_thrust_in = thrust::device_ptr<int>(dev_idata);

            thrust::sort(dev_thrust_in, dev_thrust_in + n);

            cudaMemcpy(odata, dev_idata, n * sizeof(int), cudaMemcpyDeviceToHost);

            timer().endGpuTimer();
        }
    }
}
