#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"
#include <device_launch_parameters.h>

#define blockSize 128
int* dev_bufferA;
int* dev_bufferB;


namespace StreamCompaction {
	namespace Naive {
		using StreamCompaction::Common::PerformanceTimer;
		PerformanceTimer& timer()
		{
			static PerformanceTimer timer;
			return timer;
		}
		// TODO: __global__
		__global__ void kernScanIteration(int n, int* odata, int* idata, int d) {
			int index = blockIdx.x * blockDim.x + threadIdx.x;
			if (index >= n) {
				return;
			}

			if (index >= (1 << (d-1))) {
				odata[index] = idata[index - (1 << (d - 1))] + idata[index];
			}
			else {
				odata[index] = idata[index];
			}
		}

		__global__ void kernCopyExclusive(int n, int* exclusive, int* inclusive) {
			int index = blockIdx.x * blockDim.x + threadIdx.x;
			if (index >= n) {
				return;
			}

			if (index == 0) {
				exclusive[index] = 0;
			}
			else {
				exclusive[index] = inclusive[index - 1];
			}
		}

		/**
		 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
		 */
		void scan(int n, int* odata, const int* idata) {
			if (n == 0) {
				return;
			}

			cudaMalloc((void**)&dev_bufferA, n * sizeof(int));
			cudaMalloc((void**)&dev_bufferB, n * sizeof(int));
			cudaMemcpy(dev_bufferA, idata, n * sizeof(int), cudaMemcpyHostToDevice);

			timer().startGpuTimer();
			int log2n = ilog2ceil(n);
			dim3 fullBlocksPerGrid((n  + blockSize - 1) / blockSize);
			
			for (int d = 1; d <= log2n; d++) {
				kernScanIteration << <fullBlocksPerGrid, blockSize >> > (n, dev_bufferB, dev_bufferA, d);

				if (d < log2n) {
					int* tempPtr = dev_bufferB;
					dev_bufferB = dev_bufferA;
					dev_bufferA = tempPtr;
				}

			}
			//the inclusive scan is stored in bufferB
			kernCopyExclusive << <fullBlocksPerGrid, blockSize >> > (n, dev_bufferA, dev_bufferB);
			timer().endGpuTimer();

			cudaMemcpy(odata, dev_bufferA, n * sizeof(int), cudaMemcpyDeviceToHost);
			cudaFree(dev_bufferB);
			cudaFree(dev_bufferA);

			
		}
	}
}
