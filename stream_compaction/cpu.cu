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
            // Exclusive scan
            odata[0] = 0;
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
            int osize = 0;
            for (int i = 0; i < n; ++i)
            {
                if (idata[i] != 0)
                {
                  odata[osize] = idata[i];
                  osize++;
                }
            }
            timer().endCpuTimer();
            return osize;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            int* tmp = new int[n];
            tmp[0] = 0;
            timer().startCpuTimer();
            // TODO
            // Check if corresponding elements meet criteria
            for (int i = 0; i < n; ++i)
            {
                odata[i] = idata[i] ? 1 : 0;
            }

            // Run exclusive scan
            tmp[0] = 0;
            for (int i = 1; i < n; ++i)
            {
                tmp[i] = odata[i - 1] + tmp[i - 1];
            }
            int size = tmp[n - 1];

            // Scatter
            for (int i = 0; i < n - 1; ++i)
            {
                if (tmp[i] != tmp[i + 1])
                {
                    odata[tmp[i]] = idata[i];
                }
            }
            timer().endCpuTimer();
            delete[] tmp;
            return size;
        }
    }
}
