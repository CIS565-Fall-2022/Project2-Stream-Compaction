#include <cstdio>
#include "cpu.h"

#include "common.h"

#include <iostream>

int INCOMPACT = 0;
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
            if (!INCOMPACT) {
                timer().startCpuTimer();
                // TODO
                odata[0] = 0;
                for (int i = 1; i < n; i++)
                    odata[i] = odata[i - 1] + idata[i - 1];
                timer().endCpuTimer();
            }
            else {
                odata[0] = 0;
                for (int i = 1; i < n; i++)
                    odata[i] = odata[i - 1] + idata[i - 1];
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
            int count = 0;
            for (int i = 0; i < n; i++) {
                if (idata[i] != 0) {
                    odata[count] = idata[i];
                    count ++ ;
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
            INCOMPACT = 1;
            int* copy = new int[n];
            int* scanResult = new int[n];
            timer().startCpuTimer();
            // TODO
            //copy data from idata to temporary copy buffer
            for (int i = 0; i < n; i++) {
                copy[i] = idata[i] == 0? 0 : 1;
            }
            // run exclusive scan on temporary array
            scan(n, scanResult, copy);
            //scatter
            int count = 0;
            for (int i = 0; i < n; i++) {
                if (copy[i]) {
                    count = scanResult[i];
                    odata[count] = idata[i];
                }
            }

            timer().endCpuTimer();
            delete[] copy;
            delete[] scanResult;
            return count+1;

        }
    }
}
