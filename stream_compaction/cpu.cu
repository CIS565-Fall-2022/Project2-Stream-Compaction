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

		//helper function for stream compaction to remove timer error
		void cpuScan(int n, int* odata, const int* idata) {
			if (n > 0) {
				odata[0] = 0;
				
				int prevSum = 0;
				for (int i = 1; i < n; i++) {
					odata[i] = idata[i - 1] + prevSum;
					prevSum = odata[i];
				}
			}
		}

		/**
		 * CPU scan (prefix sum).
		 * For performance analysis, this is supposed to be a simple for loop.
		 * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
		 */
		void scan(int n, int* odata, const int* idata) {
			timer().startCpuTimer();
			// TODO
			cpuScan(n, odata, idata);
			timer().endCpuTimer();
		}

		/**
		 * CPU stream compaction without using the scan function.
		 *
		 * @returns the number of elements remaining after compaction.
		 */
		int compactWithoutScan(int n, int* odata, const int* idata) {
			timer().startCpuTimer();
			// TODO
			int trueCounts = 0;
			for (int i = 0; i < n; i++) {
				if (idata[i] != 0) {
					odata[trueCounts] = idata[i];
					trueCounts++;
				}
			}
			timer().endCpuTimer();
			return trueCounts;
		}

		/**
		 * CPU stream compaction using scan and scatter, like the parallel version.
		 *
		 * @returns the number of elements remaining after compaction.
		 */
		int compactWithScan(int n, int* odata, const int* idata) {
			timer().startCpuTimer();
			int* boolArr = new int[n];
			int* scanArr = new int[n];

			for (int i = 0; i < n; i++) {
				boolArr[i] = idata[i] == 0 ? 0 : 1;
			}

			cpuScan(n, scanArr, boolArr);

			int trueCounts = 0;
			for (int i = 0; i < n; i++) {
				if (boolArr[i] == 1) {
					trueCounts++;
					odata[scanArr[i]] = idata[i];
					
				}
			}
			delete[] boolArr;
			delete[] scanArr;
			timer().endCpuTimer();
			return trueCounts;
		}
	}
}
