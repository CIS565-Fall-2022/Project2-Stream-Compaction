#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"
#include "efficient.h"

#include <device_launch_parameters.h>
#include <device_functions.h>
#include <thrust/device_ptr.h>

#define blockSize 128
#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        int* dev_idata;
        int* dev_odata;
        int* dev_unpadded_idata;

        // TODO: __global__
        __global__ void kernScan(int n, int offset, int* odata, const int* idata) {
          int index = threadIdx.x + (blockIdx.x * blockDim.x);
          if (index >= n) {
            return;
          }

          if (index >= offset) {
            odata[index] = idata[index - offset] + idata[index];
          }
          else {
            odata[index] = idata[index];
          }
        }

        // Pad the data with 1 zero at the beginning
        // And enough zeroes at the end
        // odata should be buffer of size paddedLength = the next power of two after and including (n + 1)
        __global__ void kernShiftAndPadInput(int n, int paddedLength, int* odata, int* idata) {
          int index = threadIdx.x + (blockIdx.x * blockDim.x);
          if (index == 0) {
            odata[index] = 0;
          }
          else if (index <= n) {
            odata[index] = idata[index - 1];
          }
          else if (index < paddedLength) {
            odata[index] = 0;
          }
          else {
            return;
          }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            // TODO
            
            // Add 1 because we're going to offset by a zero for exclusive scan
            int exponent = ilog2ceil(n + 1);
            int paddedLength = pow(2, exponent);
            // Input and output should be padded by 1s and 0s
            cudaMalloc((void**)&dev_unpadded_idata, n * sizeof(int));
            cudaMalloc((void**)&dev_idata, paddedLength * sizeof(int));
            cudaMalloc((void**)&dev_odata, paddedLength * sizeof(int));
            checkCUDAErrorWithLine("cudaMalloc failed");

            cudaMemcpy(dev_unpadded_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAErrorWithLine("memcpy idata failed!");

            timer().startGpuTimer();

            dim3 fullBlocksPerGrid((paddedLength + blockSize - 1) / blockSize);

            kernShiftAndPadInput<<<fullBlocksPerGrid, blockSize>>>
              (n, paddedLength, dev_idata, dev_unpadded_idata);

            //printCudaArray(paddedLength, dev_idata);

            for (int d = 1; d <= exponent; ++d) {
              int offset = pow(2, d - 1);
              kernScan << < fullBlocksPerGrid, blockSize >> > (paddedLength, offset, dev_odata, dev_idata);

              // Needed to make sure we don't launch next iteration while prev is running
              cudaDeviceSynchronize();

              std::swap(dev_odata, dev_idata);
            }

            timer().endGpuTimer();

            // Now result buffer is in dev_idata
            // We only need the first n elements of the total paddedLength elements
            cudaMemcpy(odata, dev_idata, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAErrorWithLine("memcpy back to odata failed");

            cudaFree(dev_idata);
            cudaFree(dev_odata);
            cudaFree(dev_unpadded_idata);
        }

        // oTrueData[index] is 1 if idata[index] has bit 1 at position i
        // oFalseData is just inverted oTrueData
        // bit mask is of the form (in binary) 000...1...00
        __global__ void kernGetPaddedBoolArray
          (int n, int paddedLength, int bitMask, int* falsesData, int* idata) {
          int index = threadIdx.x + (blockIdx.x * blockDim.x);
          if (index < n) {
            falsesData[index] = ((bitMask & idata[index]) == 0) ? 1 : 0;
          }
          else if (index < paddedLength) {
            // Since we pad at the end, we need to pad end elements with bit = 1 (aka. falses = 0)
            falsesData[index] = 0;
          }
        }

        __global__ void kernGetScatterAddresses(int n, int *odata, const int* falsesData, const int *falsesScanData, int totalFalses) {
          int index = threadIdx.x + (blockIdx.x * blockDim.x);
          if (index >= n) {
            return;
          }
          odata[index] = falsesData[index] == 1 
            ? falsesScanData[index]
            : index - falsesScanData[index] + totalFalses; // the "true" scan data offset by totalFalses
        }

        __global__ void kernScatter(int n, int* odata, const int* idata, const int* scatterAddresses) {
          int index = threadIdx.x + (blockIdx.x * blockDim.x);
          if (index >= n) {
            return;
          }
          odata[scatterAddresses[index]] = idata[index];
        }

        // Let's not worry about 2's complement for now...
        // Assume we only have positive integers
        void radixSort(int n, int numBits, int* odata, const int* idata) {
          int* dev_idata, * dev_odata; // two buffers needed because scatter step has race conditions
          int *dev_falsesData, * dev_falsesScanData, * dev_scatterAddresses;

          int exponent = ilog2ceil(n);
          int paddedLength = pow(2, exponent);
          dim3 fullBlocksPerGridUnpadded((n + blockSize - 1) / blockSize);
          dim3 fullBlocksPerGridPadded((paddedLength + blockSize - 1) / blockSize);

          cudaMalloc((void**)&dev_idata, n * sizeof(int));
          cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

          cudaMalloc((void**)&dev_odata, n * sizeof(int));

          cudaMalloc((void**)&dev_falsesData, paddedLength * sizeof(int));
          cudaMalloc((void**)&dev_falsesScanData, paddedLength * sizeof(int));
          cudaMalloc((void**)&dev_scatterAddresses, n * sizeof(int)); 

          int bitmask = 1;
          for (int i = 0; i < numBits; ++i) {
            // need indices from 0... paddedLength - 1
            kernGetPaddedBoolArray << < fullBlocksPerGridPadded, blockSize >> >
              (n, paddedLength, bitmask, dev_falsesData, dev_idata);
            
            cudaMemcpy(dev_falsesScanData, dev_falsesData, paddedLength * sizeof(int), cudaMemcpyDeviceToDevice);
            Efficient::scanImpl(paddedLength, dev_falsesScanData);

            // Calculate total falses = dev_falsesScanData[n-1] + dev_falsesData[n-1]
            // TODO: just make it a function
            int lastElementIsIncluded, lastBoolScanVal, totalFalses;
            cudaMemcpy(&lastElementIsIncluded, dev_falsesData + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&lastBoolScanVal, dev_falsesScanData + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            totalFalses = lastElementIsIncluded + lastBoolScanVal;
            
            kernGetScatterAddresses << <fullBlocksPerGridUnpadded, blockSize >> >
              (n, dev_scatterAddresses, dev_falsesData, dev_falsesScanData, totalFalses);

            kernScatter<<<fullBlocksPerGridUnpadded, blockSize>>>(n, dev_odata, dev_idata, dev_scatterAddresses);

            cudaDeviceSynchronize();

            //printf("MEOW total falses = %d\n", totalFalses);
            //printCudaArray(n, dev_falsesData);
            //printCudaArray(n, dev_falsesScanData);
            //printCudaArray(n, dev_scatterAddresses);
            //printCudaArray(n, dev_odata);

            bitmask = bitmask << 1; // eg. ...0010 => ...0100
            std::swap(dev_odata, dev_idata); // dev_idata always has latest
          }

          cudaMemcpy(odata, dev_idata, n * sizeof(int), cudaMemcpyDeviceToHost);

          cudaFree(dev_idata);
          cudaFree(dev_odata);
          cudaFree(dev_falsesData);
          cudaFree(dev_falsesScanData);
          cudaFree(dev_scatterAddresses);
        }
    }
}
