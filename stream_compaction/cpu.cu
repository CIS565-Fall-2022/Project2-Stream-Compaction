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
            int num_elements = 0;
            for (int i = 0; i < n; ++i) {
                if (idata[i] != 0) {
                    odata[num_elements] = idata[i];
                    ++num_elements;
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
            int* temp_array = new int[n];

            timer().startCpuTimer();
            
            int num_elements = 0;
            // GENERATE CONDITION VALUES
            for (int i = 0; i < n; ++i) {
                if (idata[i] == 0) {
                    odata[i] = 0;
                }
                else {
                    odata[i] = 1;
                }
            }

            // SCAN
            temp_array[0] = 0;
            for (int i = 1; i < n; ++i) {
                temp_array[i] = temp_array[i - 1] + odata[i - 1];
            }

            // SCATTER
            for (int i = 0; i < n; ++i) {
                if (odata[i] == 1) {
                    odata[temp_array[i]] = idata[i];
                    ++num_elements;
                }
            }
            
            timer().endCpuTimer();

            delete[] temp_array;

            return num_elements;
        }

        /**
         * CPU radix sort implementation
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to sort.
         */
        void radixSort(int n, int* odata, const int* idata) {
            // not the greatest cpu implementation of radix sort (quite memory intensive), but functional

            int* temp_array_0 = new int[n * 2];
            int* temp_array_1 = new int[n * 2];

            for (int i = 0; i < n; ++i) {
                temp_array_0[i] = idata[i];
            }

            int index_left_partition = 0;
            int index_right_partition = 0;
            int num_left_partition = n;
            int num_right_partition = 0;

            timer().startCpuTimer();

            for (int b = 0; b < sizeof(int) * 8; ++b) {
                // handle left partition
                for (int i = 0; i < num_left_partition; ++i) {
                    if (!(temp_array_0[i] & (1 << b))) {
                        temp_array_1[index_left_partition] = temp_array_0[i];
                        index_left_partition++;
                    }
                    else {
                        temp_array_1[index_right_partition + n] = temp_array_0[i];
                        index_right_partition++;
                    }
                }

                //handle right partition
                for (int i = n; i < n + num_right_partition; ++i) {
                    if (!(temp_array_0[i] & (1 << b))) {
                        temp_array_1[index_left_partition] = temp_array_0[i];
                        index_left_partition++;
                    }
                    else {
                        temp_array_1[index_right_partition + n] = temp_array_0[i];
                        index_right_partition++;
                    }
                }

                int* temp = temp_array_0;
                temp_array_0 = temp_array_1;
                temp_array_1 = temp;

                num_left_partition = index_left_partition;
                num_right_partition = index_right_partition;
                index_left_partition = 0;
                index_right_partition = 0;
            }



            timer().endCpuTimer();

            // handle left partition
            int odata_index = 0;
            for (int i = 0; i < num_left_partition; ++i) {
                odata[odata_index] = temp_array_0[i];
                odata_index++;
            }

            //handle right partition
            for (int i = n; i < n + num_right_partition; ++i) {
                odata[odata_index] = temp_array_0[i];
                odata_index++;
            }

            delete[] temp_array_0;
            delete[] temp_array_1;
        }

        /**
         * CPU radix sort implementation using std::stable_sort
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to sort.
         */
        void stdSort(int n, int* odata, const int* idata) {
            std::vector<int> vect_idata(idata, idata + n);

            timer().startCpuTimer();
            std::stable_sort(vect_idata.begin(), vect_idata.end());
            timer().endCpuTimer();

            for (int i = 0; i < n; ++i) {
                odata[i] = vect_idata[i];
            }
        }
    }
}
