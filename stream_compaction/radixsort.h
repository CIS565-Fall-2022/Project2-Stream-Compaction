#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace RadixSort {
        StreamCompaction::Common::PerformanceTimer& timer();

        void sort(int* out, const int* in, int n);
        void sortShared(int* out, const int* in, int n);
    }
}