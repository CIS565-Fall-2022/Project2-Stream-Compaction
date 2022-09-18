#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace Radix {
        StreamCompaction::Common::PerformanceTimer& timer();

        void radixSort(int n, int* odata, const int* idata);
        void thrustSort(int n, int* odata, const int* idata);
    }
}
