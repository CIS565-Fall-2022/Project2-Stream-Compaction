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
            int prev = 0;
            for (int i = 0; i < n; ++i) {
                odata[i] = prev;
                prev += idata[i];
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
            int out_ptr = 0;
            for (int i = 0; i < n; ++i) {
                if (idata[i] != 0) {
                    odata[out_ptr] = idata[i];
                    ++out_ptr;
                }
            }
            timer().endCpuTimer();
            return out_ptr;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            int* binary = new int[n];
            int* scanOut = new int[n];
            for (int i = 0; i < n; ++i) {
                if (idata[i] != 0) {
                    binary[i] = 1;
                }
                else {
                    binary[i] = 0;
                }
            }
            //Scan code copied
            int prev = 0;
            for (int i = 0; i < n; ++i) {
                scanOut[i] = prev;
                prev += binary[i];
            }
            int count = 0;
            for (int i = 0; i < n; ++i) {
                if (idata[i] != 0) {
                    odata[scanOut[i]] = idata[i];
                    ++count;
                }
            }
            timer().endCpuTimer();
            return count;
        }
    }
}
