#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace Radix_Sort {
        StreamCompaction::Common::PerformanceTimer& timer();

        int checkDigit(int num, int whichDigit);

        void radix_sort_cpu(int n, int* odata, const int* idata);
    }
}
