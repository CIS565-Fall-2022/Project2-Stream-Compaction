#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace Efficient {
        StreamCompaction::Common::PerformanceTimer& timer();

        void scan(int n, int *odata, const int *idata);

        int compact(int n, int *odata, const int *idata);

        void RadixSort(int n, int* odata, const int* idata);
        void scanWithSharedMemory(int n, int* odata, const int* idata);
        void scanWithoutTimer(int n, int* odata, const int* idata);
    }
}
