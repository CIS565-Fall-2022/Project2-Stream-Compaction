#include "common.h"

void checkCUDAErrorFn(const char *msg, const char *file, int line) {
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess == err) {
        return;
    }

    fprintf(stderr, "CUDA error");
    if (file) {
        fprintf(stderr, " (%s:%d)", file, line);
    }
    fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
}


namespace StreamCompaction {
    namespace Common {

        /**
         * Maps an array to an array of 0s and 1s for stream compaction. Elements
         * which map to 0 will be removed, and elements which map to 1 will be kept.
         */
        __global__ void kernMapToBoolean(int n, int *bools, const int *idata) {

            int thread_num = threadIdx.x + (blockIdx.x * blockDim.x);
            if (thread_num >= n) {
                return;
            }

            if (idata[thread_num] == 0) {
                bools[thread_num] = 0;
            }
            else {
                bools[thread_num] = 1;
            }
        }

        /**
         * Performs scatter on an array. That is, for each element in idata,
         * if bools[idx] == 1, it copies idata[idx] to odata[indices[idx]].
         */
        __global__ void kernScatter(int n, int *odata,
                const int *idata, const int *bools, const int *indices) {

            int thread_num = threadIdx.x + (blockIdx.x * blockDim.x);
            if (thread_num >= n) {
                return;
            }

            if (bools[thread_num] == 1) {
                odata[indices[thread_num]] = idata[thread_num];
            }
        }


        /**
         * Maps an array to an array of 0s and 1s for radix sort. Elements
         * whose value in bit b is 1 are mapped to 1, otherwise 0
         */
        __global__ void kernMapToBooleanBitwiseCheck(int n, int c, int* bools, const int* idata) {
            int thread_num = threadIdx.x + (blockIdx.x * blockDim.x);
            if (thread_num >= n) {
                return;
            }

            if (idata[thread_num] & c) {
                bools[thread_num] = 0;
            }
            else {
                bools[thread_num] = 1;
            }
        }


        __global__ void kernReverseArray(int n, int* odata_reversed, const int* odata) {
            int thread_num = threadIdx.x + (blockIdx.x * blockDim.x);
            if (thread_num >= n) {
                return;
            }

            odata_reversed[thread_num] = odata[n - thread_num - 1];
        }
    }
}
