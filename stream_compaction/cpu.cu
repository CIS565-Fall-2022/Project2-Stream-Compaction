#include <cstdio>
#include "cpu.h"

#include "common.h"

#define CPU_SIMUL_NAIVE_SCAN 0
#define CPU_SIMUL_WEFF_SCAN 0

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

        void partialScan(int* out, const int* in, int n, int stride) {
            for (int i = stride; i < n; i++) {
                out[i] = in[i] + in[i - stride];
            }
        }

        void upSweep(int* data, int n) {
            for (int stride = 2; stride <= n; stride <<= 1) {
                for (int i = stride; i <= n; i += stride) {
                    data[i - 1] += data[i - stride / 2 - 1];
                }
            }
        }

        void downSweep(int* data, int n) {
            int sum = data[n - 1];
            data[n - 1] = 0;

            for (int stride = n; stride > 1; stride >>= 1) {
                for (int i = stride; i <= n; i += stride) {
                    data[i - stride / 2 - 1] += data[i - 1];
                    std::swap(data[i - stride / 2 - 1], data[i - 1]);
                }
            }
        }

        void scanNoTimer(int n, int* odata, const int* idata) {
#if CPU_SIMUL_NAIVE_SCAN
            // simulates naive parallel scan
            int* buf = new int[n];
            buf[0] = idata[0];
            memcpy(odata, idata, n * sizeof(int));

            int stride = 1;
            while (stride < n) {
                partialScan(buf, odata, n, stride);
                memcpy(odata + stride, buf + stride, (n - stride) * sizeof(int));
                stride <<= 1;
            }
            delete[] buf;

            for (int i = n - 1; i > 0; i--) {
                odata[i] = odata[i - 1];
            }
            odata[0] = 0;
#elif CPU_SIMUL_WEFF_SCAN
            // simulates work-efficient parallel scan
            int size = ceilPow2(n);
            int* buf = new int[size];

            memcpy(buf, idata, n * sizeof(int));
            memset(buf + n, 0, (size - n) * sizeof(int));
            upSweep(buf, size);
            downSweep(buf, size);

            memcpy(odata, buf, n * sizeof(int));
            delete[] buf;
#else
            int* buf = new int[n];
            buf[0] = 0;
            for (int i = 1; i < n; i++) {
                buf[i] = buf[i - 1] + idata[i - 1];
            }
            memcpy(odata, buf, n * sizeof(int));
            delete[] buf;
#endif
        }

        void scan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // TODO
            scanNoTimer(n, odata, idata);
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
            int ptr = 0;
            for (int i = 0; i < n; i++) {
                if (idata[i]) {
                    odata[ptr++] = idata[i];
                }
            }
            timer().endCpuTimer();
            return ptr;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // TODO
            int* indices = new int[n];
            for (int i = 0; i < n; i++) {
                indices[i] = (idata[i] != 0);
            }

            scanNoTimer(n, indices, indices);

            for (int i = 0; i < n; i++) {
                if (idata[i]) {
                    odata[indices[i]] = idata[i];
                }
            }

            int size = indices[n - 1] + (idata[n - 1] != 0);
            delete[] indices;

            timer().endCpuTimer();
            return size;
        }

        void radixSort(int* out, const int* in, int n) {
            timer().startCpuTimer();
            memcpy(out, in, n * sizeof(int));

            for (uint32_t bit = 1; bit < 0x80000000u; bit <<= 1) {
                int l = 0, r = n - 1;
                while (l < r) {
                    while (out[l] & bit) {
                        l++;
                    }
                    while (!(out[r] & bit)) {
                        r--;
                    }
                    if (l < r) {
                        std::swap(out[l], out[r]);
                    }
                }
            }
            timer().endCpuTimer();
        }

        void sort(int* out, const int* in, int n) {
            timer().startCpuTimer();
            memcpy(out, in, n * sizeof(int));
            std::sort(out, out + n);
            timer().endCpuTimer();
        }
    }
}
