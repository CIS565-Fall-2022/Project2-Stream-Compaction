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
        int compactWithoutScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            int j = 0;
            for (int i = 0; i < n; ++i) {
                // If data meets criteria (not 0), add it to the output list
                if (idata[i] != 0) {
                    odata[j++] = idata[i];
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
            timer().startCpuTimer();
            int* temp = new int[n];
            int* scan = new int[n];

            // Make temporary array with 0s/1s to indicate if data meets criteria
            for (int i = 0; i < n; ++i) {
                temp[i] = (idata[i] != 0) ? 1 : 0; 
            }

            // Run exclusive scan on temporary array        
            scan[0] = 0;
            for (int j = 1; j < n; ++j) {
                scan[j] = scan[j - 1] + temp[j - 1];
            }

            // Scatter
            int elements = 0;
            for (int k = 0; k < n; ++k) {
                int meets_criteria = temp[k];
                int index = scan[k];
                if (meets_criteria) {
                    odata[index] = idata[k];
                    ++elements;
                }
            }

            delete temp, scan;
            timer().endCpuTimer();
        
            return elements;
        }
    }
}
