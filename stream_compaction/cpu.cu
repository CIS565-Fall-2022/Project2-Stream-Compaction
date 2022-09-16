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
        void scan(int n, int* odata, const int* idata) {
            timer().startCpuTimer();
            odata[0] = 0;
            for (int i = 1; i < n; ++i) {
                odata[i] = odata[i - 1] + idata[i - 1];
            }
            timer().endCpuTimer();
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int* odata, const int* idata) {
            timer().startCpuTimer();
            int oidx = 0;
            for (int i = 0; i < n; ++i) {
                if (idata[i] != 0) {
                    odata[oidx++] = idata[i];
                }
            }
            timer().endCpuTimer();
            return oidx;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int* odata, const int* idata) {
            timer().startCpuTimer();
            int* scanned = new int[n]; // (should this be included in the timer?)
            // MAP step
            for (int i = 0; i < n; ++i) {
                odata[i] = (idata[i] == 0) ? 0 : 1;
            }

            // SCAN step; scan func body (copy+pasted to avoid start/stop cpu time)
            // odata used as idata, scanned used as output
            scanned[0] = 0;
            for (int i = 1; i < n; ++i) {
                scanned[i] = scanned[i - 1] + odata[i - 1]; 
            }

            // SCATTER step
            for (int i = 0; i < n; ++i) {
              odata[scanned[i]] = idata[i];
            }
            int out = scanned[n - 1];
            delete[] scanned;
            timer().endCpuTimer();
            return out;
        }
    }
}
