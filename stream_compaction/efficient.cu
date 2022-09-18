#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kern_ComputeEfficientExclusiveScanUpSweepIteration(int N, int offset_d, int offset_d_plus_1, int* odata) {
            int thread_num = threadIdx.x + (blockIdx.x * blockDim.x);
            if (thread_num >= N) {
                return;
            }

            if (thread_num % offset_d_plus_1 == 0) {
                odata[thread_num + offset_d_plus_1 - 1] += odata[thread_num + offset_d - 1];
            }

        }

        __global__ void kern_ComputeEfficientExclusiveScanDownSweepIteration(int N, int offset_d, int offset_d_plus_1, int* odata) {
            int thread_num = threadIdx.x + (blockIdx.x * blockDim.x);
            if (thread_num >= N) {
                return;
            }

            if (thread_num % offset_d_plus_1 == 0) {
                int t = odata[thread_num + offset_d - 1];
                odata[thread_num + offset_d - 1] = odata[thread_num + offset_d_plus_1 - 1];
                odata[thread_num + offset_d_plus_1 - 1] += t;
            }

        }

        __global__ void kern_ComputeEfficientExclusiveScanUpSweepIteration_optimized(int N, int N_over_2, int num_active_threads, int offset_d, int offset_d_plus_1, int* odata) {
            int thread_num = threadIdx.x + (blockIdx.x * blockDim.x);
            if (thread_num >= num_active_threads) {
                return;
            }

            int reversed_thread_num = (N - thread_num) - 1;
            int reversed_offset = (N - offset_d) - 1;

            if (reversed_thread_num > reversed_offset) {
                odata[reversed_thread_num] += odata[reversed_thread_num - offset_d];
            }
        }

        __global__ void kern_ComputeEfficientExclusiveScanDownSweepIteration_optimized(int N, int num_active_threads, int offset_d, int offset_d_plus_1, int* odata) {
            int thread_num = threadIdx.x + (blockIdx.x * blockDim.x);
            if (thread_num >= N) {
                return;
            }

            int reversed_thread_num = (N - thread_num) - 1;
            int reversed_offset = (N - offset_d) - 1;

            if (reversed_thread_num > reversed_offset) {
                int t = odata[reversed_thread_num - offset_d];
                odata[reversed_thread_num - offset_d] = odata[reversed_thread_num];
                odata[reversed_thread_num] += t;
            }

        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int* odata, const int* idata) {
            int* dev_odata;

            // need to pad out original array to nearest power of two size
            int old_n = n;
            n = pow(2, ilog2ceil(n));

            cudaMalloc((void**)&dev_odata, n * sizeof(int));
            cudaDeviceSynchronize();

            // copy original data to GPU
            cudaMemcpy(dev_odata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
            // pad out rest of the expanded length with 0s
            cudaMemset(dev_odata + (old_n), 0, sizeof(int) * (n - old_n));
            cudaDeviceSynchronize();

            timer().startGpuTimer();

            int numBlocks = (n + blockSize - 1) / blockSize;

            int num_iterations = ilog2ceil(n);

            // UPSWEEP
            for (int d = 0; d <= num_iterations - 1; ++d) {
                // compute upsweep iteration d
                kern_ComputeEfficientExclusiveScanUpSweepIteration << < numBlocks, blockSize >> > (n, pow(2, d), pow(2, d + 1), dev_odata);
                cudaDeviceSynchronize();
            }

            // set last element in array to 0
            cudaMemset(dev_odata + n - 1, 0, sizeof(int));
            cudaDeviceSynchronize();

            // DOWNSWEEP
            for (int d = num_iterations - 1; d >= 0; --d) {
                // compute downsweep iteration d
                kern_ComputeEfficientExclusiveScanDownSweepIteration << < numBlocks, blockSize >> > (n, pow(2, d), pow(2, d + 1), dev_odata);
                cudaDeviceSynchronize();
            }

            timer().endGpuTimer();

            // copy over array from device to output array on cpu, copying only original length data
            cudaMemcpy(odata, dev_odata, sizeof(int) * old_n, cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();

            cudaFree(dev_odata);
            cudaDeviceSynchronize();
        }


        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         *
         * This is a version of Efficient::Scan that is optimized as part of EXTRA CREDIT in
         * Part 5: Why is My GPU Approach So Slow?
         */
        void scan_optimized(int n, int* odata, const int* idata) {
            /*int* dev_odata;

            // need to pad out original array to nearest power of two size
            int old_n = n;
            n = pow(2, ilog2ceil(n));

            cudaMalloc((void**)&dev_odata, n * sizeof(int));
            cudaDeviceSynchronize();

            // copy original data to GPU
            cudaMemcpy(dev_odata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
            // pad out rest of the expanded length with 0s
            cudaMemset(dev_odata + (old_n), 0, sizeof(int) * (n - old_n));
            cudaDeviceSynchronize();

            timer().startGpuTimer();

            int numBlocks = (n + blockSize - 1) / blockSize;

            int num_iterations = ilog2ceil(n);

            int num_active_threads = n;

            // UPSWEEP
            for (int d = num_iterations; d > 0; --d) {
                // compute upsweep iteration d
                num_active_threads /= 2;
                numBlocks = (num_active_threads + blockSize - 1) / blockSize;
                //std::cout << num_active_threads << std::endl;
                int reversed_thread_num = (n - 0) - 1;
                int reversed_offset = (n - num_active_threads) - 1;
                //std::cout << "fe: " << reversed_thread_num << " " << num_active_threads << " used_index: " << reversed_thread_num - num_active_threads << std::endl;
                kern_ComputeEfficientExclusiveScanUpSweepIteration_optimized << < numBlocks, blockSize >> > (n, n / 2, num_active_threads, num_active_threads, pow(2, d + 1), dev_odata);
                cudaDeviceSynchronize();
            }
            
            // set last element in array to 0
            cudaMemset(dev_odata + n - 1, 0, sizeof(int));
            cudaDeviceSynchronize();

            numBlocks = (n + blockSize - 1) / blockSize;

            num_active_threads = 1;

            // DOWNSWEEP
            for (int d = 0; d < num_iterations; ++d) {
                // compute downsweep iteration d
                
                numBlocks = (num_active_threads + blockSize - 1) / blockSize;
                kern_ComputeEfficientExclusiveScanDownSweepIteration_optimized <<< numBlocks, blockSize >>> (n, num_active_threads, num_active_threads, pow(2, d + 1), dev_odata);
                cudaDeviceSynchronize();
                num_active_threads *= 2;
            }
            
            timer().endGpuTimer();

            // copy over array from device to output array on cpu, copying only original length data
            cudaMemcpy(odata, dev_odata, sizeof(int) * old_n, cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();

            cudaFree(dev_odata);
            cudaDeviceSynchronize();*/
        }


        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         *
         * This is a version of Efficient::Scan that is optimized using shared memory as part of EXTRA CREDIT in
         * Part 6: GPU Scan Using Shared Memory && Hardware Optimization
         */
        void scan_sharedMem(int n, int* odata, const int* idata) {

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
            int* dev_idata;
            int* dev_odata;
            int* dev_boolean_map;
            int* dev_indices;

            // need to pad out original array to nearest power of two size
            int old_n = n;
            n = pow(2, ilog2ceil(n));

            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            cudaMalloc((void**)&dev_odata, n * sizeof(int));
            cudaMalloc((void**)&dev_boolean_map, n * sizeof(int));
            cudaMalloc((void**)&dev_indices, n * sizeof(int));
            cudaDeviceSynchronize();

            // copy original data to GPU
            cudaMemcpy(dev_idata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
            // pad out rest of the expanded length with 0s
            cudaMemset(dev_idata + (old_n), 0, sizeof(int) * (n - old_n));

            cudaDeviceSynchronize();

            timer().startGpuTimer();

            int numBlocks = (n + blockSize - 1) / blockSize;

            int num_iterations = ilog2ceil(n);

            ////////////////////////////////////////////
            // MAP TO BOOLEAN

            StreamCompaction::Common::kernMapToBoolean << < numBlocks, blockSize >> > (n, dev_boolean_map, dev_idata);
            cudaDeviceSynchronize();

            ////////////////////////////////////////////

            // copy data from boolean map to index array for scan
            cudaMemcpy(dev_indices, dev_boolean_map, sizeof(int) * n, cudaMemcpyDeviceToDevice);
            cudaDeviceSynchronize();
            // pad out rest of the expanded length with 0s
            cudaMemset(dev_indices + (old_n), 0, sizeof(int) * (n - old_n));

            ////////////////////////////////////////////
            // SCAN

            // UPSWEEP
            for (int d = 0; d <= num_iterations - 1; ++d) {
                // compute upsweep iteration d
                kern_ComputeEfficientExclusiveScanUpSweepIteration << < numBlocks, blockSize >> > (n, pow(2, d), pow(2, d + 1), dev_indices);
                cudaDeviceSynchronize();
            }

            // set last element in array to 0
            cudaMemset(dev_indices + n - 1, 0, sizeof(int));
            cudaDeviceSynchronize();

            // DOWNSWEEP
            for (int d = num_iterations - 1; d >= 0; --d) {
                // compute downsweep iteration d
                kern_ComputeEfficientExclusiveScanDownSweepIteration << < numBlocks, blockSize >> > (n, pow(2, d), pow(2, d + 1), dev_indices);
                cudaDeviceSynchronize();
            }

            ////////////////////////////////////////////

            ////////////////////////////////////////////
            // SCATTER

            StreamCompaction::Common::kernScatter << < numBlocks, blockSize >> > (n, dev_odata, dev_idata, dev_boolean_map, dev_indices);
            cudaDeviceSynchronize();

            ////////////////////////////////////////////



            timer().endGpuTimer();

            // copy over last index value from gpu index array to local int to return
            int last_index_val = 0;
            int last_bool_val = 0;
            cudaMemcpy(&last_index_val, dev_indices + old_n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&last_bool_val, dev_boolean_map + old_n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();

            int num_elements = last_index_val;
            if (last_bool_val == 1) {
                ++num_elements;
            }

            // copy over array from device to output array on cpu, copying only original length data
            cudaMemcpy(odata, dev_odata, sizeof(int) * old_n, cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();

            cudaFree(dev_idata);
            cudaFree(dev_odata);
            cudaFree(dev_boolean_map);
            cudaFree(dev_indices);
            cudaDeviceSynchronize();
            return num_elements;
        }

        /**
         * Performs radix sort idata, storing the result into odata.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to sort.
         */
        void radixSort(int n, int* odata, const int* idata) {
            int* dev_i;
            int* dev_b;
            int* dev_e;
            int* dev_f;
            int* dev_t;
            int* dev_d;

            // need to pad out original array to nearest power of two size
            //int old_n = n;
            //n = pow(2, ilog2ceil(n));

            cudaMalloc((void**)&dev_i, n * sizeof(int));
            cudaMalloc((void**)&dev_b, n * sizeof(int));
            cudaMalloc((void**)&dev_e, n * sizeof(int));
            cudaMalloc((void**)&dev_f, n * sizeof(int));
            cudaMalloc((void**)&dev_t, n * sizeof(int));
            cudaMalloc((void**)&dev_d, n * sizeof(int));
            cudaDeviceSynchronize();

            // copy original data to GPU
            cudaMemcpy(dev_i, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
            cudaDeviceSynchronize();

            timer().startGpuTimer();

            int numBlocks = (n + blockSize - 1) / blockSize;

            for (int b = 0; b < sizeof(int) * 8; ++b) {
                ////////////////////////////////////////////
                // MAP TO BOOLEAN
                StreamCompaction::Common::kernMapToBooleanBitwiseCheck << < numBlocks, blockSize >> > (n, 1 << b, dev_e, dev_i);
                cudaDeviceSynchronize();

                ////////////////////////////////////////////
                /*
                // copy data from boolean map to index array for scan
                cudaMemcpy(dev_f, dev_e, sizeof(int) * n, cudaMemcpyDeviceToDevice);
                cudaDeviceSynchronize();
                // pad out rest of the expanded length with 0s
                cudaMemset(dev_f + (old_n), 0, sizeof(int) * (n - old_n));

                ////////////////////////////////////////////
                // SCAN

                // UPSWEEP
                for (int d = 0; d <= num_iterations - 1; ++d) {
                    // compute upsweep iteration d
                    kern_ComputeEfficientExclusiveScanUpSweepIteration << < numBlocks, blockSize >> > (n, pow(2, d), pow(2, d + 1), dev_indices);
                    cudaDeviceSynchronize();
                }

                // set last element in array to 0
                cudaMemset(dev_indices + n - 1, 0, sizeof(int));
                cudaDeviceSynchronize();

                // DOWNSWEEP
                for (int d = num_iterations - 1; d >= 0; --d) {
                    // compute downsweep iteration d
                    kern_ComputeEfficientExclusiveScanDownSweepIteration << < numBlocks, blockSize >> > (n, pow(2, d), pow(2, d + 1), dev_indices);
                    cudaDeviceSynchronize();
                }

                ////////////////////////////////////////////
                */
            }

            timer().endGpuTimer();

            // copy over array from device to output array on cpu, copying only original length data
            cudaMemcpy(odata, dev_e, sizeof(int) * n, cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();

            cudaFree(dev_i);
            cudaFree(dev_b);
            cudaFree(dev_e);
            cudaFree(dev_f);
            cudaFree(dev_t);
            cudaFree(dev_d);
            cudaDeviceSynchronize();
        }
    }
}
