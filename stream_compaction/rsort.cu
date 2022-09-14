#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "rsort.h"

namespace StreamCompaction {
    namespace RadixSort {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer() {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernSplit() {

        }

        __global__ void kernMerge() {

        }

        void sort(int n, int* out, int const* in) {

        }
    }
}