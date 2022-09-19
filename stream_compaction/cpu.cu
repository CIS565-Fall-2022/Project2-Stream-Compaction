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

        void split(int n, int bitmask, int* odata, const int* idata) {
          // write 0s to odata in 1 pass, 1s in the next pass
          int zeroCount = 0;
          for (int i = 0; i < n; ++i) {
            if ((idata[i] & bitmask) == 0) {
              odata[zeroCount] = idata[i];
              ++zeroCount; // eg. we write a total of 123 elts with 0s, then indices 0...122s are filled
            }
          }

          int oneCount = 0;
          for (int i = 0; i < n; ++i) {
            if ((idata[i] & bitmask) != 0) {
              odata[zeroCount + oneCount] = idata[i];
              ++oneCount;
            }
          }
        }

        // CPU radix sort implementation
        void radixSort(int n, int numBits, int* odata, const int* idata)
        {
          // use temporary buffers so we don't destroy idata
          int* temp_odata = (int*) malloc(n * sizeof(int));
          int* temp_odata2 = (int*)malloc(n * sizeof(int));
          if (temp_odata == NULL || temp_odata2 == NULL) {
            printf("ERROR - failed to allocate temp_odata");
            return;
          }
          memcpy(temp_odata, idata, n * sizeof(int));

          timer().startCpuTimer();

          int bitmask = 1;
          for (int i = 0; i < numBits; ++i) {
            split(n, bitmask, temp_odata2, temp_odata);
            std::swap(temp_odata2, temp_odata); // at end of loop, temp_odata always has most updated info
            bitmask <<= 1;
          }

          timer().endCpuTimer(); // end before final memcpy (which just transfers output to odata)

          memcpy(odata, temp_odata, n * sizeof(int));

          free(temp_odata);
          free(temp_odata2);
        }
    }
}
