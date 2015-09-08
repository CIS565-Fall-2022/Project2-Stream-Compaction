#pragma once

namespace StreamCompaction {
namespace Efficient {
    void scanBlock(int n, int *odata, const int *idata);

    int compactBlock(int n, int *odata, const int *idata);
}
}
