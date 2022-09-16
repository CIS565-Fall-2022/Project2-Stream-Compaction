#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace Efficient {
        StreamCompaction::Common::PerformanceTimer& timer();

        void devScanInPlace(int* devData, int size);
        void devBlockScanInPlaceShared(int* devData, int* devBlockSum, int size, int blockSize);
        void devScanInPlaceShared(int* devData, int size);
        void devScannedBlockAdd(int* devData, int* devBlockSum, int n, int blockSize);

        void scan(int n, int *odata, const int *idata);
        void scanShared(int* out, const int* in, int n, int blockSize);

        int compact(int* out, const int* in, int n);
        int compactShared(int* out, const int* in, int n);
    }
}
