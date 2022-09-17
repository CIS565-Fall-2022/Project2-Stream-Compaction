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

        __global__ void KernNaiveScan(int n,int d,int* odata,const int* idata)
        {
            //for all k in parallel
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n)
            {
                return;
            }
            //offset: 2^d
            // 2^(offset-1)
            int d_offset = 1 << (d - 1);

            int beginIndex = index - d_offset;
            int prevData = beginIndex < 0 ? 0 : idata[beginIndex];
            odata[index] = idata[index] + prevData;

        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */

        void scan(int n, int *odata, const int *idata) {
            int blockSize = 256;
            dim3 BlocksPergrid(n + blockSize - 1 / blockSize);
            //This need to be parallel
            int* dev_idata;
            int* dev_odata;
            //allocate memory
            cudaMalloc((void**)&dev_idata, sizeof(int) * n);
            checkCUDAError("cudaMalloc dev_idata failed!");
            cudaMalloc((void**)&dev_odata, sizeof(int) * n);
            checkCUDAError("cudaMalloc dev_odata failed!");
            //Copy memory from CPU to gpu
            cudaMemcpy(dev_idata,idata,(n)*sizeof(int),cudaMemcpyHostToDevice);
            cudaMemcpy(dev_odata, idata, (n) * sizeof(int), cudaMemcpyHostToDevice);
            cudaDeviceSynchronize();
            //From host to devicw
            int log2n = ilog2ceil(n);

            timer().startGpuTimer();
            // TODO
            for (int d = 1; d <= log2n; d++)
            {
                
                KernNaiveScan << <BlocksPergrid, blockSize >> > (n,d,dev_odata,dev_idata);
                cudaDeviceSynchronize();
                //ping pong buffers
                int *dev_temp = dev_idata;
                dev_idata = dev_odata;
                dev_odata = dev_temp;
            }
            timer().endGpuTimer();
            //Exclusive scan, so need right shift.

            //copy back to host
            cudaMemcpy(odata , dev_idata, (n) * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy odata failed!");
            cudaFree(dev_idata);
            cudaFree(dev_odata);
        }
    }
}
