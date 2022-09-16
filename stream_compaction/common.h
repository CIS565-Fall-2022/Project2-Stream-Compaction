#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <iostream>
#include "intellisense.h"

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)

#define blockSize 32
// convenience macro
#define ALLOC(name, size) if(cudaMalloc((void**)&name, (size) * sizeof(*name)) != cudaSuccess) checkCUDAError("cudaMalloc " ## #name ## " failed!")
#define MEMSET(name, val, size) if(cudaMemset(name, val, size) != cudaSuccess) checkCUDAError("cudaMemset " ## #name ## " failed!")
#define FREE(name) if(cudaFree(name) != cudaSuccess) checkCUDAError("cudaFree " ## #name ## " failed!")
#define H2D(dev_name, name, size) if(cudaMemcpy(dev_name, name, (size) * sizeof(*name), cudaMemcpyHostToDevice) != cudaSuccess) checkCUDAError("cudaMemcpy from " ## #name ## " failed!")
#define D2H(name, dev_name, size) if(cudaMemcpy(name, dev_name, (size) * sizeof(*name), cudaMemcpyDeviceToHost) != cudaSuccess) checkCUDAError("cudaMemcpy to " ## #name ## " failed!")
#define D2D(dev_name1, dev_name2, size) if(cudaMemcpy(dev_name1, dev_name2, (size) * sizeof(*dev_name1), cudaMemcpyDeviceToDevice) != cudaSuccess) checkCUDAError("cudaMemcpy to " ## #dev_name1 ## " failed!")

// debug helpers
#ifndef NDEBUG
#define PRINT_GPU(arr, ...) printGPU(#arr, arr, __VA_ARGS__)
#else
#define PRINT_GPU(...)
#endif // !NDEBUG

template<typename T>
static inline void printGPU(char const* name, T* dev, int n) {
    T* tmp = new T[n];
    std::cout << name << "\n";
    D2H(tmp, dev, n);
    for (int i = 0; i < n; ++i)
        std::cout << tmp[i] << " \n"[i<n-1?0:1];
    delete[] tmp;
}

template<typename T>
static inline T getGPU(T* dev, int n, int i) {
    T tmp;
    D2H(&tmp, dev+i, 1);
    return tmp;
}

/**
 * Check for CUDA errors; print and exit if there was a problem.
 */
void checkCUDAErrorFn(const char *msg, const char *file = NULL, int line = -1);

inline int ilog2(int x) {
    int lg = 0;
    while (x >>= 1) {
        ++lg;
    }
    return lg;
}

inline int ilog2ceil(int x) {
    return x == 1 ? 0 : ilog2(x - 1) + 1;
}

namespace StreamCompaction {
    namespace Common {
        __global__ void kernMapToBoolean(int n, int *bools, const int *idata);

        __global__ void kernScatter(int n, int *odata,
                const int *idata, const int *bools, const int *indices);

        enum MakePowerTwoLengthMode { HostToDevice, DeviceToDevice };
        int makePowerTwoLength(int n, int const* in, int** out, MakePowerTwoLengthMode mode);

        /**
        * This class is used for timing the performance
        * Uncopyable and unmovable
        *
        * Adapted from WindyDarian(https://github.com/WindyDarian)
        */
        class PerformanceTimer
        {
        public:
            PerformanceTimer()
            {
                cudaEventCreate(&event_start);
                cudaEventCreate(&event_end);
            }

            ~PerformanceTimer()
            {
                cudaEventDestroy(event_start);
                cudaEventDestroy(event_end);
            }

            void startCpuTimer()
            {
                if (cpu_timer_started) { throw std::runtime_error("CPU timer already started"); }
                cpu_timer_started = true;

                time_start_cpu = std::chrono::high_resolution_clock::now();
            }

            void endCpuTimer()
            {
                time_end_cpu = std::chrono::high_resolution_clock::now();

                if (!cpu_timer_started) { throw std::runtime_error("CPU timer not started"); }

                std::chrono::duration<double, std::milli> duro = time_end_cpu - time_start_cpu;
                prev_elapsed_time_cpu_milliseconds =
                    static_cast<decltype(prev_elapsed_time_cpu_milliseconds)>(duro.count());

                cpu_timer_started = false;
            }

            void startGpuTimer()
            {
                if (gpu_timer_started) { throw std::runtime_error("GPU timer already started"); }
                gpu_timer_started = true;

                cudaEventRecord(event_start);
            }

            void endGpuTimer()
            {
                cudaEventRecord(event_end);
                cudaEventSynchronize(event_end);

                if (!gpu_timer_started) { throw std::runtime_error("GPU timer not started"); }

                cudaEventElapsedTime(&prev_elapsed_time_gpu_milliseconds, event_start, event_end);
                gpu_timer_started = false;
            }

            float getCpuElapsedTimeForPreviousOperation() //noexcept //(damn I need VS 2015
            {
                return prev_elapsed_time_cpu_milliseconds;
            }

            float getGpuElapsedTimeForPreviousOperation() //noexcept
            {
                return prev_elapsed_time_gpu_milliseconds;
            }

            // remove copy and move functions
            PerformanceTimer(const PerformanceTimer&) = delete;
            PerformanceTimer(PerformanceTimer&&) = delete;
            PerformanceTimer& operator=(const PerformanceTimer&) = delete;
            PerformanceTimer& operator=(PerformanceTimer&&) = delete;

        private:
            cudaEvent_t event_start = nullptr;
            cudaEvent_t event_end = nullptr;

            using time_point_t = std::chrono::high_resolution_clock::time_point;
            time_point_t time_start_cpu;
            time_point_t time_end_cpu;

            bool cpu_timer_started = false;
            bool gpu_timer_started = false;

            float prev_elapsed_time_cpu_milliseconds = 0.f;
            float prev_elapsed_time_gpu_milliseconds = 0.f;
        };
    }
}
