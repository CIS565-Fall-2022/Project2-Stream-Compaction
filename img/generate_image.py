from cProfile import label
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    x = np.array([29, 28, 27, 26, 25, 24, 22])
    x = list(pow(2, x) / 1e+6)

    ## 2 -> NPOT
    cpu = [1045.5, 417.7, 215.3, 108.297, 53.2, 27.4, 7.6]
    cpu2 = [901.8, 420.7, 217.0, 104, 54.2, 27.9, 6.6]

    naive = [482.4, 229.5, 108.56, 50.24, 24.2, 13.6, 2.6]
    naive2 = [474.8, 225.59, 106.0, 52.4, 24.12, 12.1, 2.9]

    eff = [151.74, 74.14, 37.4, 18.21, 9.64, 5.3, 1.23]
    eff2 = [151.59, 76.6, 37.06, 18.5, 9.89, 6.4, 1.22]

    thrust = [14.6, 9.385, 3.4, 1.84, 1.024, 0.75, 0.25]
    thrust2 = [14.3, 7.4, 3.4, 1.88, 1.042, 0.66, 1.83]

    compact_cpu = [1092.2, 577.7, 288, 145.03, 72.21, 37.1, 9.11]
    compact_cpu_scan = [6129.1, 963, 584, 246.8, 117.26, 72.3, 21.7]

    compact_eff = [213.04, 101.7, 51.7, 26.02, 14.95, 6.2, 1.63]

    sort1 = [6129.18, 963.5, 584.5, 246.8, 117.2, 72.3, 21.7]
    sort2 = [1340.75, 682.7, 338.9, 164.7, 83.30, 42.5, 12.6]

    thrust_sort = [0.0061, 0.006, 0.003, 0.0026, 0.002, 0.002, 0.0018]

    fig = plt.figure()
    plt.plot(x, cpu, label="CPU Scan")
    plt.plot(x, naive, label="CUDA Naive Scan")
    plt.plot(x, eff, label="CUDA Work-Efficient Scan")
    plt.plot(x, thrust, label="CUDA Thrust Scan")

    plt.xticks([0, 100, 200, 300, 400, 500], ["0", "100M", "200M", "300M", "400M", "500M"])

    plt.legend(fontsize=15)

    plt.title("Execution Time of Scanning (Lower Is Better)")
    plt.ylabel('Time (ms)')
    plt.xlabel("Array Size (Million)")

    plt.grid(linestyle='--')

    fig = plt.figure()
    plt.plot(x, compact_cpu, label="CPU Compact")
    plt.plot(x, compact_cpu_scan, label="CPU Compact with Scan")
    plt.plot(x, compact_eff, label="CUDA Work-Efficient Compact")

    plt.xticks([0, 100, 200, 300, 400, 500], ["0", "100M", "200M", "300M", "400M", "500M"])

    plt.legend(fontsize=15)

    plt.title("Execution Time of Stream Compaction (Lower Is Better)")
    plt.ylabel('Time (ms)')
    plt.xlabel("Array Size (Million)")

    plt.grid(linestyle='--')

    plt.show()