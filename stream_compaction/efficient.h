#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace Efficient {
        StreamCompaction::Common::PerformanceTimer& timer();

        void scan(int n, int *odata, const int *idata);

        int compact(int n, int *odata, const int *idata);
    }
}

namespace RadixSort {

    void split(int n, int* odata, int* idata, int bitpos);

    void radixSort(int n, int* odata, int* idata);
}

