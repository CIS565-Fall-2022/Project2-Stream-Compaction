#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
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
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata, bool enableTimer) {
            // TODO use `thrust::exclusive_scan`
            // example: for device_vectors dv_in and dv_out:
            // thrust::exclusive_scan(dv_in.begin(), dv_in.end(), dv_out.begin());

            //int data[6] = { 1, 0, 2, 2, 1, 3 };
            //thrust::exclusive_scan(thrust::host, data, data + 6, data, 4); // in-place scan
            //// data is now {4, 5, 5, 7, 9, 10}

            thrust::device_vector<int> dInpVec(idata, idata + n);
            thrust::device_vector<int> dOutVec(n);
            thrust::host_vector<int> hOutVec(n);

            if (enableTimer) timer().startGpuTimer();

            thrust::exclusive_scan(dInpVec.begin(), dInpVec.end(), dOutVec.begin(), 0);

            if (enableTimer) timer().endGpuTimer();

            thrust::copy(dOutVec.begin(), dOutVec.end(), hOutVec.begin());
            std::memcpy(odata, hOutVec.data(), n * sizeof(int));
        }


        void sort(int n, int* odata, const int* idata, bool enableTimer) {
            thrust::device_vector<int> dInpVec(idata, idata + n);
            thrust::host_vector<int> hOutVec(n);

            if (enableTimer) timer().startGpuTimer();

            thrust::sort(dInpVec.begin(), dInpVec.end());

            if (enableTimer) timer().endGpuTimer();

            thrust::copy(dInpVec.begin(), dInpVec.end(), hOutVec.begin());
            std::memcpy(odata, hOutVec.data(), n * sizeof(int));
        }
    }
}
