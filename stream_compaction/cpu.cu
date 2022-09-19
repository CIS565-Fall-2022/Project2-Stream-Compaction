#include <iostream>
#include <cstdio>
#include <memory>
#include "cpu.h"

#include "common.h"

namespace StreamCompaction {
    namespace CPU {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        /**
         * CPU scan (prefix sum).
         * For performance analysis, this is supposed to be a simple for loop.
         * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
         */
        void scan(int n, int* odata, const int* idata, bool enableTimer) {
            if(enableTimer) timer().startCpuTimer();
            int sum = 0;
            for (int i = 0; i < n; i++) {
                odata[i] = sum;
                sum += idata[i];
            }
            if (enableTimer) timer().endCpuTimer();
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata, bool enableTimer) {
            if (enableTimer) timer().startCpuTimer();
            int count = 0;
            for (int i = 0; i < n; i++) {
                if (idata[i] != 0)
                    odata[count++] = idata[i];
            }
            if (enableTimer) timer().endCpuTimer();
            return count;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata, bool enableTimer) {
            if (enableTimer) timer().startCpuTimer();
            std::unique_ptr<int[]> booleans{ new int[n] };
            std::unique_ptr<int[]> scanResult{ new int[n] };
            for (int i = 0; i < n; i++) {
                booleans[i] = idata[i] != 0 ? 1 : 0;
            }
            scan(n, scanResult.get(), booleans.get(), false);
            for (int i = 0; i < n; i++) {
                if (booleans[i] == 1)
                    odata[scanResult[i]] = idata[i];
            }
            if (enableTimer)  timer().endCpuTimer();
            return booleans[n - 1] > 0 ? scanResult[n - 1] + 1 : scanResult[n - 1];
        }
    }
}
