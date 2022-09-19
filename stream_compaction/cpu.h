#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace CPU {
        StreamCompaction::Common::PerformanceTimer& timer();

        void scan(int n, int *odata, const int *idata, bool enableTimer = true);

        int compactWithoutScan(int n, int *odata, const int *idata, bool enableTimer = true);

        int compactWithScan(int n, int *odata, const int *idata, bool enableTimer = true);
    }
}
