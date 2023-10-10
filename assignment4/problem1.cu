// nvcc -ccbin /usr/bin/g++-10 -lineinfo -res-usage -arch=sm_80 -std=c++11 problem1.cu -o problem1

#include <cmath>
#include <cstdlib>
#include <cuda.h>
#include <iostream>
#include <sys/time.h>

const uint64_t N = (64);
#define THRESHOLD (0.000001)

using std::cerr;
using std::cout;
using std::endl;

#define cudaCheckError(ans)                                                                        \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort)
      exit(code);
  }
}

// TODO: Edit the function definition as required
__global__ void kernel1() {}

// TODO: Edit the function definition as required
__global__ void kernel2() {}

// TODO: Edit the function definition as required
__host__ void stencil() {}

__host__ void check_result(const double* w_ref, const double* w_opt, const uint64_t size) {
  double maxdiff = 0.0, this_diff = 0.0;
  int numdiffs = 0;

  for (uint64_t i = 0; i < size; i++) {
    for (uint64_t j = 0; j < size; j++) {
      for (uint64_t k = 0; k < size; k++) {
        this_diff = w_ref[i + N * j + N * N * k] - w_opt[i + N * j + N * N * k];
        if (std::fabs(this_diff) > THRESHOLD) {
          numdiffs++;
          if (this_diff > maxdiff) {
            maxdiff = this_diff;
          }
        }
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

void print_mat(double* A) {
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      for (int k = 0; k < N; ++k) {
        printf("%lf,", A[i * N * N + j * N + k]);
      }
      printf("      ");
    }
    printf("\n");
  }
}

double rtclock() { // Seconds
  struct timezone Tzp;
  struct timeval Tp;
  int stat;
  stat = gettimeofday(&Tp, &Tzp);
  if (stat != 0) {
    cout << "Error return from gettimeofday: " << stat << "\n";
  }
  return (Tp.tv_sec + Tp.tv_usec * 1.0e-6);
}

int main() {
  uint64_t SIZE = N * N * N;

  double clkbegin = rtclock();
  stencil();
  double clkend = rtclock();
  double cpu_time = clkend - clkbegin;
  cout << "Stencil time on CPU: " << cpu_time * 1000 << " msec" << endl;

  cudaError_t status;
  cudaEvent_t start, end;
  // TODO: Fill in kernel1
  // TODO: Adapt check_result() and invoke
  float kernel_time;
  cudaEventElapsedTime(&kernel_time, start, end);
  std::cout << "Kernel 1 time (ms): " << kernel_time << "\n";

  // TODO: Fill in kernel2
  // TODO: Adapt check_result() and invoke
  cudaEventElapsedTime(&kernel_time, start, end);
  std::cout << "Kernel 2 time (ms): " << kernel_time << "\n";

  // TODO: Free memory

  return EXIT_SUCCESS;
}
