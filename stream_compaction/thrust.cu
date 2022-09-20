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

            int* dev_idata, * dev_odata;
            
            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            checkCUDAError("Error during cudaMalloc dev_idata");

            cudaMalloc((void**)&dev_odata, n * sizeof(int));
            checkCUDAError("Error during cudaMalloc dev_odata");

            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("Error during cudaMemcpy idata --> dev_idata");

            thrust::device_ptr<int> dev_thrust_idata = thrust::device_ptr<int>(dev_idata);
            thrust::device_ptr<int> dev_thrust_odata = thrust::device_ptr<int>(dev_odata);

            timer().startGpuTimer();


            thrust::exclusive_scan(dev_thrust_idata, dev_thrust_idata + n, dev_thrust_odata);
            

            // TODO use `thrust::exclusive_scan`
            // example: for device_vectors dv_in and dv_out:
            // thrust::exclusive_scan(dv_in.begin(), dv_in.end(), dv_out.begin());
            timer().endGpuTimer();

            cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("Error during cudaMemcpy dev_odata --> odata");

            cudaFree(dev_idata);
            cudaFree(dev_odata);
        }
    }
}
