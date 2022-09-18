#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace Efficient {
        StreamCompaction::Common::PerformanceTimer& timer();

        void scan(int n, int *odata, const int *idata);

        void scanImpl(int paddedLength, int* dev_data);

        int compact(int n, int *odata, const int *idata);
    }
}
