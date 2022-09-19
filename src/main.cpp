/**
 * @file      main.cpp
 * @brief     Stream compaction test program
 * @authors   Kai Ninomiya
 * @date      2015
 * @copyright University of Pennsylvania
 */

#include <cstdio>
#include <stream_compaction/cpu.h>
#include <stream_compaction/naive.h>
#include <stream_compaction/efficient.h>
#include <stream_compaction/thrust.h>
#include "testing_helpers.hpp"

 // feel free to change the size of array
 // Non-Power-Of-Two


int main(int argc, char* argv[]) {
    // Scan tests
    int test_start = 13;
    int num_tests = 14;

    float* cpu_scan_pow2 = new float[num_tests];
    float* cpu_scan_non_pow2 = new float[num_tests];
    float* naive_scan_pow2 = new float[num_tests];
    float* naive_scan_non_pow2 = new float[num_tests];
    float* efficient_scan_pow2 = new float[num_tests];
    float* efficient_scan_non_pow2 = new float[num_tests];
    float* thrust_scan_pow2 = new float[num_tests];
    float* thrust_scan_non_pow2 = new float[num_tests];

    float* cpu_compact_pow2_no_scan = new float[num_tests];
    float* cpu_compact_non_pow2_no_scan = new float[num_tests];
    float* cpu_compact_pow2_scan = new float[num_tests];
    float* efficient_compact_pow2 = new float[num_tests];
    float* efficient_compact_non_pow2 = new float[num_tests];

    for (int i = test_start; i < test_start + num_tests; ++i) {
        int SIZE = 1 << i;
        int NPOT = SIZE - 3;
        int* a = new int[SIZE];
        int* b = new int[SIZE];
        int* c = new int[SIZE];
        genArray(SIZE - 1, a, 50);  // Leave a 0 at the end to test that edge case
        a[SIZE - 1] = 0;
        printArray(SIZE, a, true);

        // initialize b using StreamCompaction::CPU::scan you implement
        // We use b for further comparison. Make sure your StreamCompaction::CPU::scan is correct.
        // At first all cases passed because b && c are all zeroes.
        zeroArray(SIZE, b);
        StreamCompaction::CPU::scan(SIZE, b, a);
        cpu_scan_pow2[i - test_start] = StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation();

        zeroArray(SIZE, c);
        StreamCompaction::CPU::scan(NPOT, c, a);
        cpu_scan_non_pow2[i - test_start] = StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation();

        zeroArray(SIZE, c);
        StreamCompaction::Naive::scan(SIZE, c, a);
        naive_scan_pow2[i - test_start] = StreamCompaction::Naive::timer().getGpuElapsedTimeForPreviousOperation();

        zeroArray(SIZE, c);
        StreamCompaction::Naive::scan(NPOT, c, a);
        naive_scan_non_pow2[i - test_start] = StreamCompaction::Naive::timer().getGpuElapsedTimeForPreviousOperation();

        zeroArray(SIZE, c);
        StreamCompaction::Efficient::scan(SIZE, c, a);
        efficient_scan_pow2[i - test_start] = StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation();
        //printArray(SIZE, c, true);
        //printCmpResult(SIZE, b, c);

        zeroArray(SIZE, c);
        StreamCompaction::Efficient::scan(NPOT, c, a);
        efficient_scan_non_pow2[i - test_start] = StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation();

        zeroArray(SIZE, c);
        StreamCompaction::Thrust::scan(SIZE, c, a);
        thrust_scan_pow2[i - test_start] = StreamCompaction::Thrust::timer().getGpuElapsedTimeForPreviousOperation();

        zeroArray(SIZE, c);
        StreamCompaction::Thrust::scan(NPOT, c, a);
        thrust_scan_non_pow2[i - test_start] = StreamCompaction::Thrust::timer().getGpuElapsedTimeForPreviousOperation();

        // Compaction tests

        genArray(SIZE - 1, a, 4);  // Leave a 0 at the end to test that edge case
        a[SIZE - 1] = 0;

        int count, expectedCount, expectedNPOT;

        // initialize b using StreamCompaction::CPU::compactWithoutScan you implement
        // We use b for further comparison. Make sure your StreamCompaction::CPU::compactWithoutScan is correct.
        zeroArray(SIZE, b);
        count = StreamCompaction::CPU::compactWithoutScan(SIZE, b, a);
        cpu_compact_pow2_no_scan[i - test_start] = StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation();
        expectedCount = count;

        zeroArray(SIZE, c);
        count = StreamCompaction::CPU::compactWithoutScan(NPOT, c, a);
        cpu_compact_non_pow2_no_scan[i - test_start] = StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation();
        //printArray(count, c, true);
        expectedNPOT = count;

        zeroArray(SIZE, c);
        count = StreamCompaction::CPU::compactWithScan(NPOT, c, a);
        cpu_compact_pow2_scan[i - test_start] = StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation();
        //printArray(count, c, true);
        //printCmpLenResult(count, expectedNPOT, b, c);

        zeroArray(SIZE, c);
        count = StreamCompaction::Efficient::compact(SIZE, c, a);
        efficient_compact_pow2[i - test_start] = StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation();

        zeroArray(SIZE, c);
        count = StreamCompaction::Efficient::compact(NPOT, c, a);
        efficient_compact_non_pow2[i - test_start] = StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation();


        delete[] a;
        delete[] b;
        delete[] c;
    }

    std::cout << "cpu scan pow2" << std::endl;
    for (int i = 0; i < num_tests; ++i) {
        std::cout << cpu_scan_pow2[i] << std::endl;
    }

    std::cout << "cpu scan non pow2" << std::endl;
    for (int i = 0; i < num_tests; ++i) {
        std::cout << cpu_scan_non_pow2[i] << std::endl;
    }

    std::cout << "thrust scan pow2" << std::endl;
    for (int i = 0; i < num_tests; ++i) {
        std::cout << thrust_scan_pow2[i] << std::endl;
    }

    std::cout << "thrust scan non pow2" << std::endl;
    for (int i = 0; i < num_tests; ++i) {
        std::cout << thrust_scan_non_pow2[i] << std::endl;
    }

    std::cout << "naive scan pow2" << std::endl;
    for (int i = 0; i < num_tests; ++i) {
        std::cout << naive_scan_pow2[i] << std::endl;
    }

    std::cout << "naive scan non pow2" << std::endl;
    for (int i = 0; i < num_tests; ++i) {
        std::cout << naive_scan_non_pow2[i] << std::endl;
    }

    std::cout << "efficient scan pow2" << std::endl;
    for (int i = 0; i < num_tests; ++i) {
        std::cout << efficient_scan_pow2[i] << std::endl;
    }

    std::cout << "efficient scan non pow2" << std::endl;
    for (int i = 0; i < num_tests; ++i) {
        std::cout << efficient_scan_non_pow2[i] << std::endl;
    }

    std::cout << "cpu compact pow2 no scan" << std::endl;
    for (int i = 0; i < num_tests; ++i) {
        std::cout << cpu_compact_pow2_no_scan[i] << std::endl;
    }

    std::cout << "cpu compact non pow2 no scan" << std::endl;
    for (int i = 0; i < num_tests; ++i) {
        std::cout << cpu_compact_non_pow2_no_scan[i] << std::endl;
    }

    std::cout << "cpu compact pow2 scan" << std::endl;
    for (int i = 0; i < num_tests; ++i) {
        std::cout << cpu_compact_pow2_scan[i] << std::endl;
    }

    std::cout << "efficient compact pow2" << std::endl;
    for (int i = 0; i < num_tests; ++i) {
        std::cout << efficient_compact_pow2[i] << std::endl;
    }

    std::cout << "efficient compact non pow2" << std::endl;
    for (int i = 0; i < num_tests; ++i) {
        std::cout << efficient_compact_non_pow2[i] << std::endl;
    }

    system("pause"); // stop Win32 console from closing on exit
}
