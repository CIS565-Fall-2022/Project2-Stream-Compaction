#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace CPU {
        StreamCompaction::Common::PerformanceTimer& timer();

        void scan(int n, int *odata, const int *idata);

        int compactWithoutScan(int n, int *odata, const int *idata);

        int compactWithScan(int n, int *odata, const int *idata);

        void radixSort(int n, int* odata, const int* idata);

        void stdSort(int n, int* odata, const int* idata);
    }
}
