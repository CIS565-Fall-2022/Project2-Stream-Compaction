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
            int pos = 0;
            for (int i = 0; i < n; i++) {
                if (idata[i] != 0) {
                    odata[pos++] = idata[i];
                }
            }
            timer().endCpuTimer();
            return pos;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            int *flags = new int[n];
            int *sum = new int[n];
            int cnt = 0;
            sum[0] = 0;
            timer().startCpuTimer();
            // TODO
            for (int i = 0; i < n; i++) {
                flags[i] = (idata[i] == 0 ? 0 : 1);
            }
            // scan (prefix sum)
            for (int i = 1; i < n; i++) {
                sum[i] = sum[i - 1] + flags[i];
            }
            // stream compaction
            for (int i = 0; i < n; i++) {
                if (flags[i] == 1) {
                    odata[sum[i]] = idata[i];
                }
            }
            timer().endCpuTimer();
            cnt = sum[n - 1] + 1;
            delete[] flags;
            delete[] sum;
            return cnt;
        }
    }
}
