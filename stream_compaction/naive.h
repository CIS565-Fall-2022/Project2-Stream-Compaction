#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace Naive {
        StreamCompaction::Common::PerformanceTimer& timer();
        __global__ void kernScan(int n, int* odata, const int* data);

        void scan(int n, int *odata, const int *idata);
    }
}
