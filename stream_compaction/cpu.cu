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
            odata[0] = 0;
            for (int i = 1; i < n; i++) {
                odata[i] = idata[i-1] + odata[i - 1];
            }
            timer().endCpuTimer();
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // TODO
            int count = 0;
            for (int i = 0; i < n; i++) {
                //do something to know if its true or false
                if (idata[i] != 0) {
                    odata[count] = idata[i];
                    count++;
                }
            }
            timer().endCpuTimer();
            return count;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // TODO
            int* odataCompact = new int[n];
            int* odataScan = new int[n];
            int odataCount = 0;
            for (int i = 0; i < n; i++) {
                odataScan[i] = idata[i] ? 1 : 0;
                if (i > 0 && odataScan[i-1]) {
                    odataCompact[i] = odataCompact[i-1] + 1;
                }
                else {
                    odataCompact[i] = i > 0 ? odataCompact[i - 1] : 0;
                }
                if (odataScan[i]) {
                    odataCount++;
                    odata[odataCompact[i]] = idata[i];
                }
            }
            timer().endCpuTimer();
            return odataCount;
        }
    }
}
