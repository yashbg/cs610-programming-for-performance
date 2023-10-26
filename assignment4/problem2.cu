// nvcc -ccbin /usr/bin/g++-10 -lineinfo -res-usage -arch=sm_80 -std=c++14 problem2.cu -o problem2

#include <cmath>
#include <cstdlib>
#include <cuda.h>
#include <iostream>
#include <sys/time.h>
#include <algorithm>
#include <chrono>
#include <thrust/scan.h>

const uint64_t N = 1 << 9;
#define THRESHOLD (0.000001)

using std::cerr;
using std::cout;
using std::endl;
using std::chrono::duration_cast;
using HR = std::chrono::high_resolution_clock;
using HRTimer = HR::time_point;
using std::chrono::nanoseconds;

#define cudaCheckError(ans)                                                                        \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort)
      exit(code);
  }
}

void print_array(int *arr) {
  for (int i = 0; i < N; ++i) {
    cout << arr[i] << " ";
  }
  cout << endl;
}

__host__ void check_result(const float* w_ref, const float* w_opt, const uint64_t size) {
  float maxdiff = 0.0, this_diff = 0.0;
  int numdiffs = 0;

  for (uint64_t i = 0; i < size; i++) {
    this_diff = std::fabs(w_ref[i] - w_opt[i]);
    if (this_diff > THRESHOLD) {
      numdiffs++;
      if (this_diff > maxdiff) {
        maxdiff = this_diff;
      }
    }
  }

  if (numdiffs > 0) {
    cout << numdiffs << " Diffs found over THRESHOLD " << THRESHOLD << "; Max Diff = " << maxdiff
         << endl;
  } else {
    cout << "No differences found between base and test versions\n";
  }
}

int main() {
  int *array = static_cast<int *>(malloc(N * sizeof(int)));
  std::fill(array, array + N, 1);

  int *ref_res = static_cast<int *>(malloc(N * sizeof(int)));
  std::fill(ref_res, ref_res + N, 0);

  HRTimer start = HR::now();
  thrust::exclusive_scan(array, array + N, ref_res);
  HRTimer end = HR::now();
  auto duration = duration_cast<nanoseconds>(end - start).count();
  cout << "Thrust time (ns): " << duration << endl;

  int *cuda_res = static_cast<int *>(malloc(N * sizeof(int)));
  std::fill(cuda_res, cuda_res + N, 0);

  start = HR::now();
  // TODO: call kernel
  end = HR::now();
  auto duration = duration_cast<nanoseconds>(end - start).count();
  cout << "CUDA time (ns): " << duration << endl;

  return EXIT_SUCCESS;
}
