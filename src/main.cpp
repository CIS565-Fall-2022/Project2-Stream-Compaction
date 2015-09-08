/**
 * @file      main.cpp
 * @brief     Stream compaction test program
 * @authors   Kai Ninomiya
 * @date      2015
 * @copyright University of Pennsylvania
 */

#include <cstdio>
#include <functional>
#include <random>
#include <stream_compaction/cpu.h>
#include <stream_compaction/naive.h>
#include <stream_compaction/efficient.h>
#include <stream_compaction/thrust.h>
#include "testing_helpers.hpp"

int main(int argc, char* argv[]) {
    const int SIZE = 1 << 8;
    const int NPOT = SIZE - 3;
    int a[SIZE], b[SIZE], c[SIZE];

    // Scan tests

    printf("\n");
    printf("*******************************\n");
    printf("** SCAN TESTS (SINGLE-BLOCK) **\n");
    printf("*******************************\n");

    genArray(SIZE, a, 50);
    printArray(SIZE, a, true);

    zeroArray(SIZE, b);
    printDesc("cpu scan, power-of-two");
    StreamCompaction::CPU::scan(SIZE, b, a);
    printArray(SIZE, b, true);

    zeroArray(SIZE, c);
    printDesc("cpu scan, non-power-of-two");
    StreamCompaction::CPU::scan(NPOT, c, a);
    printArray(NPOT, b, true);
    printCmpResult(NPOT, b, c);

    zeroArray(SIZE, c);
    printDesc("naive scan, power-of-two");
    StreamCompaction::Naive::scan(SIZE, c, a);
    //printArray(SIZE, c, true);
    printCmpResult(SIZE, b, c);

    zeroArray(SIZE, c);
    printDesc("naive scan, non-power-of-two");
    StreamCompaction::Naive::scan(NPOT, c, a);
    //printArray(SIZE, c, true);
    printCmpResult(NPOT, b, c);

    zeroArray(SIZE, c);
    printDesc("work-efficient scan, power-of-two");
    StreamCompaction::Efficient::scanBlock(SIZE, c, a);
    //printArray(SIZE, c, true);
    printCmpResult(SIZE, b, c);

    zeroArray(SIZE, c);
    printDesc("work-efficient scan, non-power-of-two");
    StreamCompaction::Efficient::scanBlock(NPOT, c, a);
    //printArray(NPOT, c, true);
    printCmpResult(NPOT, b, c);

    zeroArray(SIZE, c);
    printDesc("thrust scan, power-of-two");
    StreamCompaction::Thrust::scan(SIZE, c, a);
    //printArray(SIZE, c, true);
    printCmpResult(SIZE, b, c);

    zeroArray(SIZE, c);
    printDesc("thrust scan, non-power-of-two");
    StreamCompaction::Thrust::scan(NPOT, c, a);
    //printArray(NPOT, c, true);
    printCmpResult(NPOT, b, c);

    printf("\n");
    printf("********************************************\n");
    printf("** STREAM COMPACTION TESTS (SINGLE-BLOCK) **\n");
    printf("********************************************\n");

    // Compaction tests

    genArray(SIZE, a, 4);
    printArray(SIZE, a, true);

    int count, expectedCount, expectedNPOT;

    zeroArray(SIZE, b);
    printDesc("cpu compact without scan, power-of-two");
    count = StreamCompaction::CPU::compactWithoutScan(SIZE, b, a);
    expectedCount = count;
    printArray(count, b, true);
    printCmpLenResult(count, expectedCount, b, b);

    zeroArray(SIZE, c);
    printDesc("cpu compact without scan, non-power-of-two");
    count = StreamCompaction::CPU::compactWithoutScan(NPOT, c, a);
    expectedNPOT = count;
    printArray(count, c, true);
    printCmpLenResult(count, expectedNPOT, b, c);

    zeroArray(SIZE, c);
    printDesc("cpu compact with scan");
    count = StreamCompaction::CPU::compactWithScan(SIZE, c, a);
    expectedCount = count;
    printArray(count, c, true);
    printCmpLenResult(count, expectedCount, b, c);

    zeroArray(SIZE, c);
    printDesc("work-efficient compact, power-of-two");
    count = StreamCompaction::Efficient::compactBlock(SIZE, c, a);
    //printArray(count, c, true);
    printCmpLenResult(count, expectedCount, b, c);

    zeroArray(SIZE, c);
    printDesc("work-efficient compact, non-power-of-two");
    count = StreamCompaction::Efficient::compactBlock(NPOT, c, a);
    //printArray(count, c, true);
    printCmpLenResult(count, expectedNPOT, b, c);
}
