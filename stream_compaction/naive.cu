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

        /*
        __global__ void scan(float *g_odata, float *g_idata, int n) {   
            extern __shared__ float temp[]; // allocated on invocation    
            int thid = threadIdx.x;   
            int pout = 0, pin = 1;   
            // Load input into shared memory.    
            // This is exclusive scan, so shift right by one    
            // and set first element to 0   
            temp[pout*n + thid] = (thid > 0) ? g_idata[thid-1] : 0;  
            __syncthreads();   
            for (int offset = 1; offset < n; offset *= 2)   
            {     
                pout = 1 - pout; // swap double buffer indices     
                pin = 1 - pout;     
                if (thid >= offset)       
                    temp[pout*n+thid] += temp[pin*n+thid - offset];     
                else       
                    temp[pout*n+thid] = temp[pin*n+thid];
                    __syncthreads();  
            }   
            g_odata[thid] = temp[pout*n+thid]; // write output 
        } 
        
        */


        __global__ void kernNaiveScan(int n, int exp2dminus1, int* odata, const int* idata) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index >= n)
            {
                return;
            }

            if (index >= exp2dminus1)
            {
                odata[index] = idata[index - exp2dminus1] + idata[index];
            }
            else
            {
                odata[index] = idata[index];
            }
        }

        __global__ void kernNaiveShift(int n, int* idata, int* odata)
        {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index >= n)
            {
                return;
            }
            else if (index == 0)
            {
                odata[index] = 0;
            }
            else
            {
                odata[index] = idata[index - 1];
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {

            dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);

            int* dev_buffer1 = nullptr;
            int* dev_buffer2 = nullptr;
            cudaMalloc((void**)&dev_buffer1, n * sizeof(int));
            cudaMalloc((void**)&dev_buffer2, n * sizeof(int));
            cudaMemcpy(dev_buffer1, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            timer().startGpuTimer();
            // TODO

            for (int d = 1; d <= ilog2ceil(n);++d)
            {
                int exp2D = 1 << (d - 1);
                kernNaiveScan << <fullBlocksPerGrid, blockSize >> > (n, exp2D, dev_buffer2, dev_buffer1);
                int* temp = dev_buffer1;
                dev_buffer1 = dev_buffer2;
                dev_buffer2 = temp;
            }

            // we made a inclusive scan using naive scan, we need to shift all entries one to the right
            kernNaiveShift << <fullBlocksPerGrid, blockSize >> > (n, dev_buffer1, dev_buffer2);

            timer().endGpuTimer();
            cudaMemcpy(odata, dev_buffer2, n * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(dev_buffer1);
            cudaFree(dev_buffer2);
        }
    }
}
