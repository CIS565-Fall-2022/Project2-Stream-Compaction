#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

#define blockSize 128 

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        __global__ void kernUpsweep(int n, int depth, int* odata) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            int path = pow(2, depth);
            if (index >= n || index % (2*path) != 0 ||(index + 2*path-1) >= n) {
                return;
            }
            odata[index + 2*path - 1] += odata[index + path - 1];
            return;
        }

        __global__ void kernDownsweep(int n, int depth, int* odata) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            int path = pow(2, depth);
            if (index >= n || index % (2 * path) != 0 || (index+2*path-1) >= n) {
                return;
            }
            int saveVal = odata[index + path - 1];
            odata[index + path - 1] = odata[index + 2 * path - 1];
            odata[index + 2 * path - 1] += saveVal;
        }

        void fillArrayToPower2(int* odataMax, const int* idata, int maxN, int n) {
            cudaMalloc((void**)&odataMax, maxN * sizeof(int));
            cudaMemcpy(odataMax, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemset(odataMax + n, 0, (maxN - n) * sizeof(int));
        }
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            //fill array and initialization
            int maxN = pow(2, ilog2ceil(n));
            dim3 blockDim((maxN + blockSize - 1) / blockSize);
            int* odataMax;
            fillArrayToPower2(odataMax, idata, maxN, n);

            timer().startGpuTimer();
            // TODO
            //Upsweep
            for (int i = 0; i < ilog2ceil(n); i++) {
                kernUpsweep <<<blockDim, blockSize >> > (n, i, odataMax);
            }
            //Downsweep
            odataMax[maxN - 1] = 0;
            for (int i = ilog2ceil(n) - 1; i > 0; i--) {
                kernDownsweep << <blockDim, blockSize >> > (n, i, odataMax);
            }
            timer().endGpuTimer();

            cudaMemcpy(odata, odataMax, n*sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(odataMax);
        }

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int *odata, const int *idata) {
            //initialization
            int maxN = pow(2, ilog2ceil(n));
            dim3 blockDim((n + blockSize - 1) / blockSize);
            int* odataMax, *oBoll, * oScan;
            fillArrayToPower2(odataMax, idata, maxN, n);
            cudaMalloc((void**)&oBoll, maxN * sizeof(int));
            cudaMalloc((void**)&oScan, maxN * sizeof(int));

            timer().startGpuTimer();
            // TODO
            //first create temp array
            StreamCompaction::Common::kernMapToBoolean << <blockDim, blockSize >> > (n, oBoll, idata);
            cudaMemcpy(oScan, oBoll, maxN * sizeof(int), cudaMemcpyDeviceToDevice);
            //Upsweep
            for (int i = 0; i < ilog2ceil(n); i++) {
                kernUpsweep << <blockDim, blockSize >> > (n, i, oScan);
            }
            //Downsweep
            oScan[n - 1] = 0;
            for (int i = ilog2ceil(n) - 1; i > 0; i--) {
                kernDownsweep << <blockDim, blockSize >> > (n, i, oScan);
            }
            //scatter
            StreamCompaction::Common::kernScatter << <blockDim, blockSize >> > (n, odata, idata, odataMax, oScan);
            timer().endGpuTimer();


            cudaFree(oBoll);
            cudaFree(oScan);
            cudaFree(odataMax);
            return -1;//how do we get the final index?
        }
    }
}
