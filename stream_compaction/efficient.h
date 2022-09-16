#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace Efficient {
        StreamCompaction::Common::PerformanceTimer& timer();

        void devScanInPlace(int* devData, int size);
        void devScanInPlaceShared(int* devData, int size, int blockSize);

        void scan(int n, int *odata, const int *idata);
        void scanWithSharedMemory(int* out, const int* in, int n);

        int compact(int n, int *odata, const int *idata);
    }
}
