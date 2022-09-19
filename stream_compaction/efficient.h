#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace Efficient {
        StreamCompaction::Common::PerformanceTimer& timer();
        __global__ void kernScan(int n, int* data);
        void scan(int n, int *odata, const int *idata);

        int compact(int n, int *odata, const int *idata);
    }
}
