#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"


#define blockSize 128
#define THREAD_OPTIMIZATION 1;

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernUpSweep(int n, int d, int* data){
            int k = threadIdx.x + (blockIdx.x * blockDim.x);
            if (k >= n) {
                return;
            }

#if THREAD_OPTIMIZATION
            int stride = 1 << (d + 1);
            k = (k + 1) * stride - 1;
            data[k] += data[k - (stride >> 1)];
#else
            int stride = 1 << (d + 1);
            if (k % stride == 0) {
                data[k + stride - 1] += data[k + (1 << d) - 1];
            }
#endif
            



        }

        __global__ void kernDownSweep(int n, int d, int* data) {
            int k = threadIdx.x + (blockIdx.x * blockDim.x);
            if (k >= n) {
                return;
            }

            
#if THREAD_OPTIMIZATION
            int stride = 1 << (d + 1);
            k = (k + 1) * stride - 1;
            int t = data[k - (stride >> 1)];
            data[k - (stride >> 1)] = data[k];
            data[k] += t;
#else
            int stride = 1 << (d + 1);
            int pow_2 = 1 << d;
            if (k % stride == 0) {
                int t = data[k + pow_2 - 1];
                data[k + pow_2 - 1] = data[k + stride - 1];
                data[k + stride - 1] = t + data[k + stride - 1];
            }
#endif
        }

        void perfixSumScan(int size, int* idata) {
            dim3 fullBlocksPerGrid((size + blockSize - 1) / blockSize);
            int threads = size;
            for (int d = 0; d < ilog2ceil(size); d++) {
#if THREAD_OPTIMIZATION
                dim3 blocksPerGrid((threads + blockSize - 1) / blockSize);
                threads /= 2;
                kernUpSweep <<< blocksPerGrid, blockSize >> > (threads, d, idata);
                
#else
                kernUpSweep << < fullBlocksPerGrid, blockSize >> > (size, d, idata);
#endif
            }
            //set last element  to 0
            
            int val = 0;           
            cudaMemcpy(&idata[size - 1], &val, sizeof(int), cudaMemcpyHostToDevice);

            //threads is already 1
            for (int d = ilog2ceil(size) - 1; d >= 0; d--) {
#if THREAD_OPTIMIZATION
                dim3 blocksPerGrid((threads + blockSize - 1) / blockSize);
                kernDownSweep << < blocksPerGrid, blockSize >> > (threads, d, idata);
                threads *= 2;
#else
                kernDownSweep << < fullBlocksPerGrid, blockSize >> > (size, d, idata);
#endif
            }
        }
        /**uyb
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            int* dev_data;
                 
            //for non-pow2
            int size = 1 << ilog2ceil(n);
            cudaMalloc((void**)&dev_data, size * sizeof(int));
            cudaMemcpy(dev_data, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
           
            
            timer().startGpuTimer();     
            // TODO
            perfixSumScan(size, dev_data);   
            timer().endGpuTimer();
            
            cudaMemcpy(odata, dev_data, sizeof(int) * n, cudaMemcpyDeviceToHost);

            cudaFree(dev_data);
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
            dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);

            int* dev_bool;
            int* dev_idata;
            int* dev_odata;
            int* dev_scanResult;

            int count = 0;
            int lastBool = 0;

            int size = 1 << ilog2ceil(n);

            cudaMalloc((void**)&dev_bool, n * sizeof(int));
            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            cudaMalloc((void**)&dev_odata, n * sizeof(int));
            cudaMalloc((void**)&dev_scanResult, size * sizeof(int));

           

            cudaMemcpy(dev_idata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);

            timer().startGpuTimer();
            // TODO
            Common::kernMapToBoolean << <fullBlocksPerGrid, blockSize >> > (n, dev_bool, dev_idata);
            cudaMemcpy(dev_scanResult, dev_bool, n * sizeof(int), cudaMemcpyDeviceToDevice);
            perfixSumScan(size, dev_scanResult);
            Common::kernScatter << <fullBlocksPerGrid, blockSize >> > (n, dev_odata, dev_idata, dev_bool, dev_scanResult);
            timer().endGpuTimer();

            //last boolean is not counted in exclusive scan, if last bool is 1, need to take this into account
            //This is not shown in the slides
            cudaMemcpy(&count, dev_scanResult + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&lastBool, dev_bool + n - 1, sizeof(int), cudaMemcpyDeviceToHost);

            int length = count + lastBool;
            cudaMemcpy(odata, dev_odata, sizeof(int) * length, cudaMemcpyDeviceToHost);

            cudaFree(dev_bool);
            cudaFree(dev_idata);
            cudaFree(dev_odata);
            cudaFree(dev_scanResult);
            return length;
        }
    }
}
