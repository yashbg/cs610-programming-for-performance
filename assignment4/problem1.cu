// nvcc -ccbin /usr/bin/g++-10 -lineinfo -res-usage -arch=sm\_80 -std=c++11 190997-prob1.cu -o 190997-prob1

#include <cmath>
#include <cstdlib>
#include <cuda.h>
#include <iostream>
#include <sys/time.h>
#include <algorithm>

const uint64_t N = (64);
#define THRESHOLD (0.000001)

const uint64_t MAX_VAL = 1e6;
const int TILE_DIM = 16;
const int BLOCK_DIM_Z = 8;

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

// naive CUDA kernel
__global__ void kernel1(const float *in, float *out) {
  int block_rows = blockDim.y;
  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;
  
  for (int j = 0; j < TILE_DIM; j += block_rows) {
    if (x > 0 && x < N - 1 && y + j > 0 && y + j < N - 1 && z > 0 && z < N - 1) {
      int idx = z * N * N + (y + j) * N + x;
      out[idx] = 0.8 * (in[(z - 1) * N * N + (y + j) * N + x]
                      + in[(z + 1) * N * N + (y + j) * N + x]
                      + in[z * N * N + (y + j - 1) * N + x]
                      + in[z * N * N + (y + j + 1) * N + x]
                      + in[z * N * N + (y + j) * N + (x - 1)]
                      + in[z * N * N + (y + j) * N + (x + 1)]);
    }
  }
}

// shared memory tiling
__global__ void kernel2(const float *in, float *out) {
  __shared__ float tile[TILE_DIM * TILE_DIM * BLOCK_DIM_Z];

  int block_rows = blockDim.y;
  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;
  
  for (int j = 0; j < TILE_DIM; j += block_rows) {
    if (x > 0 && x < N - 1 && y + j > 0 && y + j < N - 1 && z > 0 && z < N - 1) {
      int idx = threadIdx.z * TILE_DIM * TILE_DIM
              + (threadIdx.y + j) * TILE_DIM
              + threadIdx.x;
      tile[idx] = 0.8 * (in[(z - 1) * N * N + (y + j) * N + x]
                       + in[(z + 1) * N * N + (y + j) * N + x]
                       + in[z * N * N + (y + j - 1) * N + x]
                       + in[z * N * N + (y + j + 1) * N + x]
                       + in[z * N * N + (y + j) * N + (x - 1)]
                       + in[z * N * N + (y + j) * N + (x + 1)]);
    }
  }

  __syncthreads();

  for (int j = 0; j < TILE_DIM; j += block_rows) {
    if (x > 0 && x < N - 1 && y + j > 0 && y + j < N - 1 && z > 0 && z < N - 1) {
      int idx = threadIdx.z * TILE_DIM * TILE_DIM
              + (threadIdx.y + j) * TILE_DIM
              + threadIdx.x;
      out[z * N * N + (y + j) * N + x] = tile[idx];
    }
  }
}

__host__ void stencil(const float *in, float *out) {
  for (int i = 1; i < N - 1; i++) {
    for (int j = 1; j < N - 1; j++) {
      for (int k = 1; k < N - 1; k++) {
        out[i * N * N + j * N + k] = 0.8 * (in[(i - 1) * N * N + j * N + k]
                                          + in[(i + 1) * N * N + j * N + k]
                                          + in[i * N * N + (j - 1) * N + k]
                                          + in[i * N * N + (j + 1) * N + k]
                                          + in[i * N * N + j * N + (k - 1)]
                                          + in[i * N * N + j * N + (k + 1)]);
      }
    }
  }
}

__host__ void check_result(const float* w_ref, const float* w_opt, const uint64_t size) {
  float maxdiff = 0.0, this_diff = 0.0;
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

void print_mat(float* A) {
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

  float *h_in = static_cast<float *>(malloc(SIZE * sizeof(float)));
  float *h_out_serial = static_cast<float *>(malloc(SIZE * sizeof(float)));
  float *h_out = static_cast<float *>(malloc(SIZE * sizeof(float)));

  for (int i = 0; i < SIZE; i++) {
    h_in[i] = std::rand() % MAX_VAL;
  }
  std::fill_n(h_out_serial, SIZE, 0.0);
  std::fill_n(h_out, SIZE, 0.0);

  double clkbegin = rtclock();
  stencil(h_in, h_out_serial);
  double clkend = rtclock();
  double cpu_time = (clkend - clkbegin) * 1000;
  cout << "Stencil time on CPU: " << cpu_time << " msec" << endl << endl;

  // naive CUDA kernel

  float *d_in;
  cudaCheckError(cudaMalloc(&d_in, SIZE * sizeof(float)));
  float *d_out;
  cudaCheckError(cudaMalloc(&d_out, SIZE * sizeof(float)));

  cudaCheckError(cudaMemcpy(d_in, h_in, SIZE * sizeof(float),
                            cudaMemcpyHostToDevice));

  cudaEvent_t start, end;
  cudaCheckError(cudaEventCreate(&start));
  cudaCheckError(cudaEventCreate(&end));

  int block_rows = 8;
  dim3 dimBlock(TILE_DIM, block_rows, BLOCK_DIM_Z);
  dim3 dimGrid(N / TILE_DIM, N / TILE_DIM, N / BLOCK_DIM_Z);

  cudaCheckError(cudaEventRecord(start));
  kernel1<<<dimGrid, dimBlock>>>(d_in, d_out);
  cudaCheckError(cudaEventRecord(end));
  cudaCheckError(cudaEventSynchronize(end));

  cudaCheckError(cudaMemcpy(h_out, d_out, SIZE * sizeof(float),
                            cudaMemcpyDeviceToHost));
  check_result(h_out_serial, h_out, N);

  float kernel_time;
  cudaCheckError(cudaEventElapsedTime(&kernel_time, start, end));
  cout << "Kernel 1 time (ms): " << kernel_time << ", Speedup: " << cpu_time / kernel_time << endl << endl;

  // shared memory tiling

  int block_rows_arr[] = {1, 2, 4, 8};
  for (auto block_rows : block_rows_arr) {
    std::fill_n(h_out, SIZE, 0.0);

    dim3 dimBlock(TILE_DIM, block_rows, BLOCK_DIM_Z);
    dim3 dimGrid(N / TILE_DIM, N / TILE_DIM, N / BLOCK_DIM_Z);

    cudaCheckError(cudaEventRecord(start));
    kernel2<<<dimGrid, dimBlock>>>(d_in, d_out);
    cudaCheckError(cudaEventRecord(end));
    cudaCheckError(cudaEventSynchronize(end));

    cudaCheckError(cudaMemcpy(h_out, d_out, SIZE * sizeof(float),
                              cudaMemcpyDeviceToHost));
    check_result(h_out_serial, h_out, N);

    cudaCheckError(cudaEventElapsedTime(&kernel_time, start, end));
    cout << "Kernel 2 (block size = " << block_rows << ") time (ms): " << kernel_time << ", Speedup: " << cpu_time / kernel_time << endl << endl;
  }

  // pinned memory

  float *h_in_pinned;
  cudaCheckError(cudaHostAlloc(&h_in_pinned, SIZE * sizeof(float),
                               cudaHostAllocDefault));
  float *h_out_pinned;
  cudaCheckError(cudaHostAlloc(&h_out_pinned, SIZE * sizeof(float),
                               cudaHostAllocDefault));

  for (int i = 0; i < SIZE; i++) {
    h_in_pinned[i] = h_in[i];
  }
  std::fill_n(h_out_pinned, SIZE, 0.0);

  cudaCheckError(cudaMemcpy(d_in, h_in_pinned, SIZE * sizeof(float),
                            cudaMemcpyHostToDevice));

  cudaCheckError(cudaEventRecord(start));
  kernel2<<<dimGrid, dimBlock>>>(d_in, d_out);
  cudaCheckError(cudaEventRecord(end));
  cudaCheckError(cudaEventSynchronize(end));

  cudaCheckError(cudaMemcpy(h_out_pinned, d_out, SIZE * sizeof(float),
                            cudaMemcpyDeviceToHost));
  check_result(h_out_serial, h_out, N);

  cudaCheckError(cudaEventElapsedTime(&kernel_time, start, end));
  cout << "Kernel 2 (pinned memory) time (ms): " << kernel_time << ", Speedup: " << cpu_time / kernel_time << endl << endl;

  // unified virtual memory

  float *in_uvm;
  cudaCheckError(cudaMallocManaged(&in_uvm, SIZE * sizeof(float)));
  float *out_uvm;
  cudaCheckError(cudaMallocManaged(&out_uvm, SIZE * sizeof(float)));
  for (int i = 0; i < SIZE; i++) {
    in_uvm[i] = h_in[i];
  }
  std::fill_n(out_uvm, SIZE, 0.0);

  cudaCheckError(cudaEventRecord(start));
  kernel2<<<dimGrid, dimBlock>>>(in_uvm, out_uvm);
  cudaCheckError(cudaEventRecord(end));
  cudaCheckError(cudaEventSynchronize(end));

  check_result(h_out_serial, out_uvm, N);

  cudaCheckError(cudaEventElapsedTime(&kernel_time, start, end));
  cout << "Kernel 2 (uvm) time (ms): " << kernel_time << ", Speedup: " << cpu_time / kernel_time << endl << endl;

  cudaCheckError(cudaEventDestroy(start));
  cudaCheckError(cudaEventDestroy(end));
  cudaCheckError(cudaFree(d_in));
  cudaCheckError(cudaFree(d_out));
  cudaCheckError(cudaFreeHost(h_in_pinned));
  cudaCheckError(cudaFreeHost(h_out_pinned));
  cudaCheckError(cudaFree(in_uvm));
  cudaCheckError(cudaFree(out_uvm));

  free(h_in);
  free(h_out_serial);
  free(h_out);

  return EXIT_SUCCESS;
}
