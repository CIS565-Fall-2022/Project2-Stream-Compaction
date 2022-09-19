#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        // TODO: __global__
        __global__ void kernScan(const int* idata, int* odata, int n, int d) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n) {
                return;
            }

            int pow2dminus1 = 1 << d - 1;
            if (index >= pow2dminus1) {
                odata[index] = idata[index] + idata[index - pow2dminus1];
            }
            else {
                odata[index] = idata[index];
            }
        }
        // Part 5
        //__global__ void scan(float* g_odata, float* g_idata, int n) {
        //    extern __shared__ float temp[]; // allocated on invocation    
        //    int thid = threadIdx.x;
        //    int pout = 0, pin = 1;
        //    temp[pout * n + thid] = (thid > 0) ? g_idata[thid - 1] : 0;
        //    __syncthreads();
        //    for (int offset = 1; offset < n; offset *= 2) {
        //        pout = 1 - pout;
        //        // swap double buffer indices     
        //        pin = 1 - pout;
        //        if (thid >= offset)
        //            temp[pout * n + thid] += temp[pin * n + thid - offset];
        //        else
        //            temp[pout * n + thid] = temp[pin * n + thid];
        //        __syncthreads();
        //    }
        //    g_odata[thid] = temp[pout * n + thid]; // write output 
        //} 


        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int* odata, const int* idata) {
            dim3 blocksPerGrid((n + BLOCK_SIZE - 1) / BLOCK_SIZE);

            int* dev_in;
            int* dev_out;
            cudaMalloc((void**)&dev_in, n * sizeof(int));
            cudaMalloc((void**)&dev_out, n * sizeof(int));
            cudaMemcpy(dev_in, idata, sizeof(int) * n, cudaMemcpyHostToDevice);

            timer().startGpuTimer();

            for (int d = 1; d <= ilog2ceil(n); d++) {
                kernScan <<< blocksPerGrid, BLOCK_SIZE >>> (dev_in, dev_out, n, d);
                //std::swap(dev_in, dev_out);
                cudaMemcpy(dev_in, dev_out, sizeof(int) * n, cudaMemcpyDeviceToDevice);
            }
            // works fine without doing the exclusive shift here as mentioned in GOU Gem Book
            timer().endGpuTimer();

            // send the data to host 
            // shift the output by 1 for exclusive scanc
            odata[0] = 0;
            cudaMemcpy(odata + 1, dev_in, sizeof(int) * (n - 1), cudaMemcpyDeviceToHost);
            cudaFree(dev_in);
            cudaFree(dev_out);
        }
    }
}