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

        void exclusiveScan(int n, int* odata, const int* idata) {
          if (n <= 0) {
            printf("Empty array!");
            return;
          }
          odata[0] = 0; // identity
          for (int i = 0; i < n - 1; ++i) {
            odata[i + 1] = odata[i] + idata[i];
          }
        }

        /**
         * CPU scan (prefix sum).
         * For performance analysis, this is supposed to be a simple for loop.
         * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // TODO
            exclusiveScan(n, odata, idata);
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
            int outputIndex = 0;
            for (int inputIndex = 0; inputIndex < n; ++inputIndex) {
              if (idata[inputIndex] != 0) {
                odata[outputIndex] = idata[inputIndex];
                outputIndex += 1;
              }
            }
            timer().endCpuTimer();
            return outputIndex;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // TODO
            // calculate bool. array
            int* boolData = new int[n];
            if (boolData == NULL) {
              printf("Failed to allocate boolData");
              return -1;
            }
            for (int i = 0; i < n; ++i) {
              boolData[i] = (idata[i] == 0) ? 0 : 1;
            }

            int* boolScan = new int[n];
            if (boolScan == NULL) {
              printf("Failed to allocate boolScan");
              return -1;
            }
            exclusiveScan(n, boolScan, boolData);
            //printf("boolScan ");
            //for (int i = 0; i < n; ++i) {
            //  printf("%d ", boolScan[i]);
            //}

            int outIndex = -1;
            // Scatter
            for (int i = 0; i < n; ++i) {
              if (boolData[i] == 1) {
                outIndex = boolScan[i];
                odata[outIndex] = idata[i];
              }
            }
            delete[] boolData;
            delete[] boolScan;
            timer().endCpuTimer();

            return outIndex + 1;
        }
    }
}
