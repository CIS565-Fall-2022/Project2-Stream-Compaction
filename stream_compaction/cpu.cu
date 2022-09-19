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
            odata[0] = idata[0];
            for (int k = 1; k < n; k++)
            {
                odata[k] = odata[k - 1] + idata[k];
            }
            //shift
            for (int i = n - 1; i > 0; i--)
            {
                odata[i] = odata[i - 1];
            }
            odata[0] = 0;
            timer().endCpuTimer();
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            int num_elements = 0;
            for (int i = 0; i < n; i++)
            {
                if (idata[i] != 0)
                {
                    odata[num_elements] = idata[i];
                    num_elements++;
                }
            }
            timer().endCpuTimer();
            return num_elements;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            int* temp_array = new int[n];
            for (int i = 0; i < n; i++)
            {
                if (idata[i] != 0)
                {
                    temp_array[i] = 1;
                }
                else
                {
                    temp_array[i] = 0;
                }
            }
            //Scan
            odata[0] = temp_array[0];
            for (int k = 1; k < n; k++)
            {
                odata[k] = odata[k - 1] + temp_array[k];
            }
            //shift
            for (int i = n - 1; i > 0; i--)
            {
                odata[i] = odata[i - 1];
            }
            odata[0] = 0;

            //Scatter
            int num_elements = 0;
            for (int i = 0; i < n; i++)
            {
                if (temp_array[i] == 1)
                {
                    odata[odata[i]] = idata[i];
                    num_elements++;
                }
            }
            delete[] temp_array;


            timer().endCpuTimer();
            return num_elements;
        }
    }
}
