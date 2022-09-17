#include <cstdio>
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
        void scan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // TODO
            odata[0] = idata[0];
            for (size_t i = 1; i < n; i++) {
                odata[i] = odata[i - 1] + idata[i];
            }
            timer().endCpuTimer();
        }
        void scanNoTimer(int n, int* odata, const int* idata) {
            // TODO
            odata[0] = idata[0];
            for (size_t i = 1; i < n; i++) {
                odata[i] = odata[i - 1] + idata[i];
            }
        }
        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // TODO
            int j = 0;
            for (size_t i = 0; i < n; i++) {
                if (idata[i] != 0) {
                    odata[j] = idata[i];
                    j++;
                }
            }

            timer().endCpuTimer();
            return j;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            int* boolFlag = new int[n]; // temporary array 
            int* scanRes = new int[n];
            timer().startCpuTimer();
            // TODO
            for (size_t i = 0; i < n; i++)
            {
                boolFlag[i] = idata[i] == 0 ? 0 : 1;
            }
            scanNoTimer(n, scanRes, boolFlag); // odata: scan result
            

            for (size_t i = 0; i < n; i++) {
                if (boolFlag[i] == 0) continue;
                odata[scanRes[i] - 1] = idata[i];
            }

            timer().endCpuTimer();
            return scanRes[n - 1];

        }
    }
}
