#include <cstdio>
#include "cpu.h"

#include "common.h"
#define IDENTITY 0

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

            odata[0] = idata[0];
            for (int i = 1; i < n; i++) {
                odata[i] = idata[i] + odata[i-1];
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
            int currIdx = 0;

            for (int i = 0; i < n; i++) {
                if (idata[i] != 0) {
                    odata[currIdx] = idata[i];
                    currIdx++;
                }
            }
            

            timer().endCpuTimer();
            return currIdx;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            
            int* scanned = new int[n];
            //Filter
            for (int i = 0; i < n; i++) {
                (idata[i] != 0) ? scanned[i] = 1 : scanned[i] = 0;
            }

            //Exclusive Scan
            idata[0] == 0 ? scanned[0] = 0 : scanned[0] = 1;
            for (int i = 1; i < n; i++) {
                scanned[i] = scanned[i] + scanned[i - 1];
            }

            //Scatter
            int currIdx = 0;
            //odata[currIdx] = idata[0];



            //SCATTERING MUTHA FUCKAAAA REEEEE
            if (idata[0] > 0) {
                odata[0] = idata[0];
                currIdx++;
            }

            for (int i = 1; i < n; i++) {
                if (scanned[i] > scanned[i - 1]) {
                    odata[scanned[i]-1] = idata[i];
                    currIdx++;
                }
                else {
                    odata[i] = 0;
                }
            }
            
            delete[] scanned;

            timer().endCpuTimer();
            return currIdx;
        }
    }
}
