#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"
#include <device_launch_parameters.h>

#define blockSize 128
int* dev_idata;
int* dev_odata;
int* dev_scan_idata;
int* dev_boolArr;
int* dev_indexArr;

namespace StreamCompaction {
	namespace Efficient {
		using StreamCompaction::Common::PerformanceTimer;
		PerformanceTimer& timer()
		{
			static PerformanceTimer timer;
			return timer;
		}

		__global__ void kernUpSweep(int n, int* idata, int d) {
			int index = blockIdx.x * blockDim.x + threadIdx.x;
			if (index >= n) {
				return;
			}

			if (index % (1 << (d + 1)) == 0) {
				idata[index + (1 << (d + 1)) - 1] += idata[index + (1 << d) - 1];
			}
		}

		__global__ void kernSetRootNode(int n, int* idata) {
			int index = blockIdx.x * blockDim.x + threadIdx.x;
			if (index >= n) {
				return;
			}

			idata[n - 1] = 0;
		}

		__global__ void kernDownSweep(int n, int* idata, int d) {
			int index = blockIdx.x * blockDim.x + threadIdx.x;
			if (index >= n) {
				return;
			}

			if (index % ((1 << (d + 1))) == 0) {
				int t = idata[index + (1 << d) - 1];
				idata[index + (1 << d) - 1] = idata[index + (1 << (d + 1)) - 1];
				idata[index + (1 << (d + 1)) - 1] += t;
			}
		}

		//helper function for stream compaction to remove timer error
		void eff_scan(int n, int* odata, const int* idata) {
			int log2n = ilog2ceil(n);
			int sizeOfArr = 1 << log2n;

			dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);

			cudaMalloc((void**)&dev_scan_idata, n * sizeof(int));
			cudaMemcpy(dev_scan_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

			for (int d = 0; d <= log2n - 1; d++) {
				kernUpSweep << <fullBlocksPerGrid, blockSize >> > (sizeOfArr, dev_scan_idata, d);
			}

			kernSetRootNode << <1, 1 >> > (sizeOfArr, dev_scan_idata);

			for (int d = log2n - 1; d >= 0; d--) {
				kernDownSweep << <fullBlocksPerGrid, blockSize >> > (sizeOfArr, dev_scan_idata, d);
			}

			cudaMemcpy(odata, dev_scan_idata, n * sizeof(int), cudaMemcpyDeviceToHost);
			cudaFree(dev_scan_idata);
		}

		/**
		 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
		 */
		void scan(int n, int* odata, const int* idata) {

			int log2n = ilog2ceil(n);
			int sizeOfArr = 1 << log2n;

			dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);

			cudaMalloc((void**)&dev_scan_idata, n * sizeof(int));
			cudaMemcpy(dev_scan_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

			timer().startGpuTimer();
			for (int d = 0; d <= log2n - 1; d++) {
				kernUpSweep << <fullBlocksPerGrid, blockSize >> > (sizeOfArr, dev_scan_idata, d);
			}

			kernSetRootNode << <1, 1 >> > (sizeOfArr, dev_scan_idata);

			for (int d = log2n - 1; d >= 0; d--) {
				kernDownSweep << <fullBlocksPerGrid, blockSize >> > (sizeOfArr, dev_scan_idata, d);
			}
			timer().endGpuTimer();

			cudaMemcpy(odata, dev_scan_idata, n * sizeof(int), cudaMemcpyDeviceToHost);
			cudaFree(dev_scan_idata);

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
		int compact(int n, int* odata, const int* idata) {

			// TODO

			int log2n = ilog2ceil(n);
			int sizeOfArr = 1 << log2n;

			dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);
			cudaMalloc((void**)&dev_idata, n * sizeof(int));
			cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

			cudaMalloc((void**)&dev_boolArr, sizeOfArr * sizeof(int));
			cudaMalloc((void**)&dev_indexArr, sizeOfArr * sizeof(int));
			cudaMalloc((void**)&dev_odata, sizeof(int) * n);

			timer().startGpuTimer();
			StreamCompaction::Common::kernMapToBoolean << <fullBlocksPerGrid, blockSize >> > (sizeOfArr, dev_boolArr, dev_idata);

			eff_scan(n, dev_indexArr, dev_boolArr);

			StreamCompaction::Common::kernScatter << <fullBlocksPerGrid, blockSize >> > (n, dev_odata, dev_idata, dev_boolArr, dev_indexArr);
			timer().endGpuTimer();

			cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);

			int* indices = new int[n];
			int count = 0;
			cudaMemcpy(indices, dev_indexArr, n * sizeof(int), cudaMemcpyDeviceToHost);

			cudaFree(dev_idata);
			cudaFree(dev_odata);
			cudaFree(dev_boolArr);
			cudaFree(dev_indexArr);


			if (idata[n - 1] == 0) {
				count = indices[n - 1];
			}
			else {
				count = indices[n - 1] + 1;
			}

			delete[] indices;
			return count;
		}
	}
}
