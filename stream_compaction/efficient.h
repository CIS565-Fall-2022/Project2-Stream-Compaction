#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace Efficient {
        StreamCompaction::Common::PerformanceTimer& timer();

        void scan(int n, int *odata, const int *idata);

        void scanRecursion(int n, int* data, dim3 blockPerGrid);


        int compact(int n, int *odata, const int *idata);
    }
}
