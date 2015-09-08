#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
namespace Efficient {

// TODO: __global__

/**
 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
 */
void scanBlock(int n, int *odata, const int *idata) {
    // TODO
    printf("TODO\n");
}

/**
 * Performs stream compaction on idata, storing the result into odata.
 * All zeroes are discarded.
 *
 * @param n      The number of elements in idata.
 * @param odata  The array into which to store elements.
 * @param idata  The array of elements to compact.
 * @returns      The number of elements remaining after compaction.
 */
int compactBlock(int n, int *odata, const int *idata) {
    // TODO
    return -1;
}

}
}
