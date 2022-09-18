#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        void printArray(int n, int* a, bool abridged = false) {
            printf("    [ ");
            for (int i = 0; i < n; i++) {
                if (abridged && i + 2 == 15 && n > 16) {
                    i = n - 2;
                    printf("... ");
                }
                printf("%3d ", a[i]);
            }
            printf("]\n");
        }

        __global__ void kernPadZeroes(int old_n, int N, int* data) {
            int index = ((blockIdx.x * blockDim.x) + threadIdx.x);
            if (index < N && index >= old_n) {
                data[index] = 0;
            }
        }

        __global__ void kernUpSweep(int span, int N, int* data) {
            int index = ((blockIdx.x * blockDim.x) + threadIdx.x);
            if (index > N / span) {
                return;
            }
            //if (span == 1 << 15) {
            //    data[N - 1] = index * span;// +span * 2 - 1;
            //    return;
            //}
            index *= span * 2;
            if (N > index + span - 1) {
                data[index + span * 2 - 1] += data[index + span - 1];
                //data[index + span * 2 - 1] = index + span * 2 - 1;
            }
        }

        __global__ void kernDownSweep(int span, int N, int* data, bool first) {
            int index = ((blockIdx.x * blockDim.x) + threadIdx.x);
            //This should run once for N - 1
            if (index > N / span) {
                return;
            }
            if (first) {
                if (index == N - 1) {
                    data[N - 1] = 0;
                }
                __syncthreads();
            }
            index *= span * 2;
            
            if (N > index + span * 2 - 1) {
                int t = data[index + span - 1];
                data[index + span - 1] = data[index + span * 2 - 1];
                data[index + span * 2 - 1] += t;
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            
            // TODO
            int old_n = n;
            n = (int)(pow(2, ilog2ceil(n)) + .5);
            int* dev_in;
            int* temp = new int[n];
            cudaMalloc((void**)&dev_in, n * sizeof(int));
            cudaMemcpy(dev_in, idata, sizeof(int) * n, cudaMemcpyHostToDevice);

            /*cudaMemcpy(odata, dev_in, sizeof(int) * n, cudaMemcpyDeviceToHost);
            printArray(n, odata, true);*/

            int blockSize = 128;
            dim3 fullBlocks((n + blockSize - 1) / blockSize);
            timer().startGpuTimer();
            kernPadZeroes << <fullBlocks, blockSize >> > (old_n, n, dev_in);
            int span = 1;
            for (int d = 0; d <= (int)ilog2ceil(n) - 1; d++) {
                kernUpSweep << <fullBlocks, blockSize >> > (span, n, dev_in);
                checkCUDAErrorWithLine("kernUpSweep for end failed!");
                /*cudaMemcpy(odata, dev_in, sizeof(int) * n, cudaMemcpyDeviceToHost);
                printArray(n, odata, true);*/
                span *= 2;
            }
            span /= 2;

            //printf( "-----------------------------------------------------\n");
            cudaMemcpy(temp, dev_in, sizeof(int) * n, cudaMemcpyDeviceToHost);
            temp[n - 1] = 0;
            cudaMemcpy(dev_in, temp, sizeof(int) * n, cudaMemcpyHostToDevice);

            bool first = true;
            for (int d = (int)ilog2ceil(n) - 1; d >= 0; d--) {
                kernDownSweep << <fullBlocks, blockSize >> > (span, n, dev_in, first);
                /*cudaMemcpy(odata, dev_in, sizeof(int) * n, cudaMemcpyDeviceToHost);
                printArray(n, odata, true);*/
                first = false;
                span /= 2;
            }
            timer().endGpuTimer();

            cudaMemcpy(odata, dev_in, sizeof(int) * old_n, cudaMemcpyDeviceToHost);
            cudaFree(dev_in);
            delete[] temp;
        }

        __global__ void kernBitArray(int N, int* odata, int* idata) {
            int index = ((blockIdx.x * blockDim.x) + threadIdx.x);
            if (index >= N) {
                return;
            }
            if (idata[index] != 0) {
                odata[index] = 1;
            }
            else {
                odata[index] = 0;
            }
        }

        __global__ void kernFillOut(int N, int* odata, int* bitdata, int* scandata, int* idata) {
            int index = ((blockIdx.x * blockDim.x) + threadIdx.x);
            if (index >= N) {
                return;
            }
            if (bitdata[index] == 1) {
                odata[scandata[index]] = idata[index];
            }
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
            //allocate memory
            int* dev_out;
            int* dev_in;
            int* dev_bit;
            int* dev_scan;
            int* host_bit = new int[n];
            int* host_scan = new int[n];
            int* host_out = new int[n];
            cudaMalloc((void**)&dev_out, n * sizeof(int));
            cudaMalloc((void**)&dev_in, n * sizeof(int));
            cudaMalloc((void**)&dev_bit, n * sizeof(int));
            cudaMalloc((void**)&dev_scan, n * sizeof(int));

            cudaMemcpy(dev_out, odata, sizeof(int) * n, cudaMemcpyHostToDevice);
            cudaMemcpy(dev_in, idata, sizeof(int) * n, cudaMemcpyHostToDevice);

            //timer().startGpuTimer();
            // TODO
            /*cudaMemcpy(odata, dev_in, sizeof(int) * n, cudaMemcpyDeviceToHost);
            printArray(n, odata);*/

            int blockSize = 128;
            dim3 fullBlocks((n + blockSize - 1) / blockSize);
            kernBitArray << <fullBlocks, blockSize >> > (n, dev_bit, dev_in);

            cudaMemcpy(host_bit, dev_bit, sizeof(int) * n, cudaMemcpyDeviceToHost);
            //printArray(n, host_bit);
            scan(n, host_scan, host_bit);
            //printArray(n, host_scan);
            cudaMemcpy(dev_scan, host_scan, sizeof(int) * n, cudaMemcpyHostToDevice);

            kernFillOut << <fullBlocks, blockSize >> > (n, dev_out, dev_bit, dev_scan, dev_in);

            //timer().endGpuTimer();
            
            cudaMemcpy(host_out, dev_scan, sizeof(int) * n, cudaMemcpyDeviceToHost);
            int count = host_out[n - 1];
            if (host_bit[n - 1] == 1) {
                count++;
            }
            cudaMemcpy(odata, dev_out, sizeof(int) * count, cudaMemcpyDeviceToHost);
            //printArray(n, odata);

            //Free memory
            cudaFree(dev_out);
            cudaFree(dev_in);
            cudaFree(dev_bit);
            cudaFree(dev_scan);
            delete[] host_bit;
            delete[] host_scan;
            delete[] host_out;
            return count;
        }
    }
}
