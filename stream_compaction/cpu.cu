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
            // perform exclusive prefix sum
            for (int i = 0; i < n; i++) {
                if (i == 0) {
                    odata[i] = 0;
                }
                else {
                    odata[i] = odata[i - 1] + idata[i - 1];
                }
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
            // removes digits that are 0
            int oi = 0;
            for (int i = 0; i < n; i++) {
                if (idata[i] != 0) {
                    odata[oi] = idata[i];
                    oi++;
                }
            }
            timer().endCpuTimer();

            // return number of valid digits
            return oi;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // TODO
            int* tdata = new int[n];
            int* sdata = new int[n];

            int numvalid = 0;
            for (int i = 0; i < n; i++) {
                if (idata[i] != 0) {
                    tdata[i] = 1;
                }
                else {
                    // todo remove if initialize to 0 automatically
                    tdata[i] = 0;
                }
            }

            // perform scan on tdata to produce sdata
            // perform exclusive prefix sum
            for (int i = 0; i < n; i++) {
                if (i == 0) {
                    sdata[i] = 0;
                }
                else {
                    sdata[i] = sdata[i - 1] + tdata[i - 1];
                }
            }

            // scatter: iterate over tdata. if tdata is 1, write idata[i] to odata[sdata[i]]
            // where odata.size is last elem of sdata
            for (int i = 0; i < n; i++) {
                if (tdata[i] == 1) {
                    odata[sdata[i]] = idata[i];
                }
            }

            int numValid = sdata[n - 1] + tdata[n - 1];

            delete[] tdata;
            delete[] sdata;

            timer().endCpuTimer();
            return numValid;
        }
    }
}
