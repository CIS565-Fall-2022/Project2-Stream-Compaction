#include <cstdio>
#include <device_launch_parameters.h>
#include "radix_sort.h"
#include "cpu.h"
#include <iostream>

#include "common.h"

namespace StreamCompaction 
{
    namespace Radix_Sort 
    {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        int checkDigit(int num, int whichDigit)
        {
            return (num >> whichDigit) & 1;
        }

        void radix_sort_cpu(int n, int* odata, const int* idata)
        {
            int* oArray = new int[n];   // A copy of odata for scattering
            int* eArray = new int[n];
            int* fArray = new int[n];
            int* tArray = new int[n];

            // Init odata
            memcpy(odata, idata, n * sizeof(int));

            timer().startCpuTimer();

            // Sort by each digit (since the biggest num is 3, so there will be 2 digits at most) 
            for (int i = 0; i < 6; ++i)
            {
                // Save the orignal odata
                memcpy(oArray, odata, n * sizeof(int));

                // Build eArray
                for (int j = 0; j < n; ++j)
                {
                    eArray[j] = checkDigit(oArray[j], i) == 1 ? 0 : 1;
                }

                // Build fArray by scaning eArray
                CPU::doScan(n, fArray, eArray);

                // Scatter data by d
                int d;
                int totalFalses = fArray[n - 1] + eArray[n - 1];
                for (int j = 0; j < n; ++j)
                {
                    if (eArray[j] == 0) // b[j] == 1
                    {
                        d = j - fArray[j] + totalFalses; // d[j] = t[j]
                    }
                    else
                    {
                        d = fArray[j];  // d[j] = f[j]
                    }                 
                    odata[d] = oArray[j];
                }
            }

            timer().endCpuTimer();

            delete[] oArray;
            delete[] eArray;
            delete[] fArray;
            delete[] tArray;            
        }
    }
}
