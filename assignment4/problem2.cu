// nvcc -ccbin /usr/bin/g++-10 -lineinfo -res-usage -arch=sm\_80 -std=c++14 190997-prob2.cu -o 190997-prob2

#include <cmath>
#include <cstdlib>
#include <cuda.h>
#include <iostream>
#include <sys/time.h>
#include <algorithm>
#include <chrono>
#include <thrust/scan.h>

const uint64_t N = 1 << 10;
#define THRESHOLD (0.000001)
const uint64_t MAX_VAL = 1e3;

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

__global__ void kernel(const float *in, float *out) {
  __shared__ float temp[2 * N];

  int thid = threadIdx.x;

  int pout = 0;
  int pin = 1;

  temp[pout * N + thid] = (thid > 0) ? in[thid - 1] : 0;

  for (int offset = 1; offset < N; offset *= 2) {
    pout = 1 - pout;
    pin  = 1 - pout;
    __syncthreads();

    temp[pout * N + thid] = temp[pin * N + thid]; 

    if (thid >= offset) {
      temp[pout * N + thid] += temp[pin * N + thid - offset];
    }
  }

  __syncthreads();

  out[thid] = temp[pout * N + thid];
}

void thrust_prefix_sum(const float *in, float *out) {
  thrust::exclusive_scan(in, in + N, out);
}

void print_array(float *arr) {
  for (int i = 0; i < N; i++) {
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
  float *h_in = static_cast<float *>(malloc(N * sizeof(float)));
  for (int i = 0; i < N; i++) {
    h_in[i] = std::rand() % MAX_VAL;
  }

  float *ref_out = static_cast<float *>(malloc(N * sizeof(float)));
  std::fill(ref_out, ref_out + N, 0);

  HRTimer hr_start = HR::now();
  thrust_prefix_sum(h_in, ref_out);
  HRTimer hr_end = HR::now();
  float duration = duration_cast<nanoseconds>(hr_end - hr_start).count();
  cout << "Thrust time (ms): " << duration / 1000000 << endl;

  float *h_out = static_cast<float *>(malloc(N * sizeof(float)));
  std::fill(h_out, h_out + N, 0);

  float *d_in;
  cudaCheckError(cudaMalloc(&d_in, N * sizeof(float)));
  float *d_out;
  cudaCheckError(cudaMalloc(&d_out, N * sizeof(float)));

  cudaCheckError(cudaMemcpy(d_in, h_in, N * sizeof(float),
                            cudaMemcpyHostToDevice));

  cudaEvent_t start, end;
  cudaCheckError(cudaEventCreate(&start));
  cudaCheckError(cudaEventCreate(&end));

  int dimBlock = N;
  int dimGrid = 1;
  
  cudaCheckError(cudaEventRecord(start));
  kernel<<<dimGrid, dimBlock>>>(d_in, d_out);
  cudaCheckError(cudaEventRecord(end));
  cudaCheckError(cudaEventSynchronize(end));

  cudaCheckError(cudaMemcpy(h_out, d_out, N * sizeof(float),
                            cudaMemcpyDeviceToHost));
  check_result(ref_out, h_out, N);

  float kernel_time;
  cudaCheckError(cudaEventElapsedTime(&kernel_time, start, end));
  cout << "Kernel time (ms): " << kernel_time << endl;

  cudaCheckError(cudaEventDestroy(start));
  cudaCheckError(cudaEventDestroy(end));
  cudaCheckError(cudaFree(d_in));
  cudaCheckError(cudaFree(d_out));

  free(h_in);
  free(ref_out);
  free(h_out);

  return EXIT_SUCCESS;
}
