#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace Efficient {
        StreamCompaction::Common::PerformanceTimer& timer();

        void scan(int n, int *odata, const int *idata);
        void fillArray(int** odata, const int* idata, int maxN, int n);
        void scanImpl(int n, int* odata);
        int compact(int n, int *odata, const int *idata);
    }
}
