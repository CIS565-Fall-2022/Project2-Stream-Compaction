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
            int identity = 0;
            odata[0] = identity;    // exclusive scan
            for (int i = 1; i < n; ++i)
            {
                odata[i] = odata[i - 1] + idata[i - 1];
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
            int oIndex = 0;
            for (int i = 0; i < n; ++i)
            {
                if (idata[i] != 0)
                {
                    odata[oIndex] = idata[i];
                    oIndex++; 
                }
            }
            timer().endCpuTimer();
            return oIndex;  // the number of elements remaining
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            int* keepData = new int[n];
            // Step 1 and Step 2
            timer().startCpuTimer();
            int identity = 0;
            odata[0] = identity;    // exclusive scan

            for (int i = 0; i < n; ++i)
            {
                    keepData[i] = (idata[i] == 0)? 0 : 1;
                    if (i > 0)
                    odata[i] = odata[i - 1] + keepData[i - 1];

            }

            // Step 3
            int oIndex = 0;
            for (int i = 0; i < n; ++i)
            {
                if (keepData[i])
                {
                    odata[oIndex] = idata[i];
                    oIndex++;
                }
            }
            timer().endCpuTimer();
            delete[] keepData;
            return oIndex;
        }
    }
}
