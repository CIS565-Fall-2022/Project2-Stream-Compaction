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

        __global__ void kernUpSweep(int threadNeeded, int d, int* dev_idata) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            //increase1 2^(d+1), increase2 2^d
            if (index < threadNeeded) {
                int increase1 = 1 << (d + 1);
                int increase2 = 1 << d;
                int multiIdx = index * increase1;
                dev_idata[multiIdx + increase1 - 1] += dev_idata[multiIdx + increase2 - 1];
            }
        }

        __global__ void kernDownSweep(int threadNeeded, int d, int* dev_idata){
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index < threadNeeded) {
                int increase1 = 1 << (d + 1);
                int increase2 = 1 << d;
                int multiIdx = index * increase1;
                int t = dev_idata[multiIdx + increase2 - 1];
                dev_idata[multiIdx + increase2 - 1] = dev_idata[multiIdx + increase1 - 1];
                dev_idata[multiIdx + increase1 - 1] += t;
            }
        }

        __global__ void kernMapToBoolean(int n, int* temp_Arr, int* dev_idata) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index < n) {
                if (dev_idata[index] != 0) {
                    temp_Arr[index] = 1;
                }
            }
        }

        __global__ void kernScatter(int n, int* dev_tempArr, int* dev_finalArr, int* dev_idata) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index < (n - 1)) {
                int currScan = dev_tempArr[index];
                int nextScan = dev_tempArr[index + 1];
                if (currScan < nextScan) {
                    dev_finalArr[currScan] = dev_idata[index];
                }
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {

            //used to round the array sizes to the next power of two.
            int nCeil = ilog2ceil(n);
            int n2PowCeil = 1 << nCeil;

            int* dev_idata;
            cudaMalloc((void**)&dev_idata, n2PowCeil * sizeof(int));
            checkCUDAError("cudaMalloc dev_idata failed!");

            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy idata to dev_idata failed!");
            //timer start
            timer().startGpuTimer();
            if (n2PowCeil != n) {
                cudaMemset(&(dev_idata[n]), 0, (n2PowCeil - n) * sizeof(int));
                checkCUDAError("cudaMemset failed!");
            }
            
            //open n threads is enough
            // TODO
            //up-sweep
            int depth = ilog2ceil(n2PowCeil) - 1;
            for (int d = 0; d <= depth; ++d) {
                int threadNeeded = 1 << (nCeil - d - 1);
                dim3 fullBlocksPerGrid((blockSize + threadNeeded - 1) / blockSize);
                kernUpSweep << <fullBlocksPerGrid, blockSize>> > (threadNeeded, d, dev_idata);
            }
            //down-sweep
            cudaMemset(&(dev_idata[n2PowCeil -1]), 0, sizeof(int));
            for (int d = depth; d >= 0; --d) {
                int threadNeeded = 1 << (nCeil - d - 1);
                dim3 fullBlocksPerGrid((blockSize + threadNeeded - 1) / blockSize);
                kernDownSweep << <fullBlocksPerGrid, blockSize >> > (threadNeeded, d, dev_idata);
            }
            timer().endGpuTimer();


            cudaMemcpy(odata, dev_idata, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("memory dev_idata to odata failed!");

            cudaFree(dev_idata);
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

            int nCeil = ilog2ceil(n);
            int n2PowCeil = 1 << nCeil;
            int* dev_idata;
            int* dev_tempArr;
            int* dev_finalArr;

            cudaMalloc((void**)&dev_idata, n2PowCeil * sizeof(int));
            checkCUDAError("cudaMalloc dev_idata failed!");

            cudaMalloc((void**)&dev_tempArr, n2PowCeil * sizeof(int));
            checkCUDAError("cudaMalloc dev_tempArr failed!");

            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy from idata to dev_idata failed!");
            
            //start
            timer().startGpuTimer();
            if (n2PowCeil != n) {
                cudaMemset(&(dev_idata[n]), 0, (n2PowCeil - n) * sizeof(int));
                checkCUDAError("cudaMemset dev_idata failed!");
            }

            cudaMemset(dev_tempArr, 0, n2PowCeil * sizeof(int));
            checkCUDAError("cudaMemset dev_tempArr failed!");

            dim3 fullBlocksPerGrid((blockSize + n - 1) / blockSize);

            // build boolean array
            kernMapToBoolean << <fullBlocksPerGrid, blockSize >> > (n, dev_tempArr, dev_idata);
            int lastElement = idata[n - 1];

            //up-sweep
            int depth = ilog2ceil(n2PowCeil) - 1;
            for (int d = 0; d <= depth; ++d) {
                int threadNeeded = 1 << (nCeil - d - 1);
                dim3 fullBlocksPerGrid((blockSize + threadNeeded - 1) / blockSize);
                kernUpSweep << <fullBlocksPerGrid, blockSize >> > (threadNeeded, d, dev_tempArr);
            }
            //create final array based on up-sweep result
            int numOfResults;
            cudaMemcpy(&numOfResults, &(dev_tempArr[n2PowCeil - 1]), sizeof(int), cudaMemcpyDeviceToHost);
            cudaMalloc((void**)&dev_finalArr, numOfResults * sizeof(int));
            //down-sweep
            cudaMemset(&(dev_tempArr[n2PowCeil - 1]), 0, sizeof(int));
            for (int d = depth; d >= 0; --d) {
                int threadNeeded = 1 << (nCeil - d - 1);
                dim3 fullBlocksPerGrid((blockSize + threadNeeded - 1) / blockSize);
                kernDownSweep << <fullBlocksPerGrid, blockSize >> > (threadNeeded, d, dev_tempArr);
            }
            //scatter
            kernScatter << <fullBlocksPerGrid, blockSize >> > (n, dev_tempArr, dev_finalArr, dev_idata);

            timer().endGpuTimer();
            //end

            cudaMemcpy(odata, dev_finalArr, numOfResults * sizeof(int), cudaMemcpyDeviceToHost);
            if (lastElement) {
                odata[numOfResults - 1] = lastElement;
            }

            return numOfResults;
        }
    }
}
