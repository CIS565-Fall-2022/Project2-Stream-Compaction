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
            for (int k = 1; k < n; ++k) {
                odata[k] = odata[k - 1] + idata[k - 1];
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
            int resultSize = 0;
            for (int i = 0; i < n; ++i) {
                int num = idata[i];
                if (num) {
                    odata[resultSize] = num;
                    resultSize++;
                }
            }
            timer().endCpuTimer();
            return resultSize;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // TODO
            // create bool array
            for (int i = 0; i < n; ++i) {
                odata[i] = idata[i] ? 1 : 0;
            }

            // scan 
            int* scan = new int[n];
            scan[0] = 0;
            for (int i = 1; i < n; ++i) {
                scan[i] = scan[i - 1] + odata[i - 1];
            }

            // scatter
            int resultSize = 0;
            for (int i = 0; i < n; ++i) {
                if (odata[i]) {
                    odata[scan[i]] = idata[i];
                    resultSize++;
                } 
            }
            timer().endCpuTimer();
            return resultSize;
        }
    }
}
