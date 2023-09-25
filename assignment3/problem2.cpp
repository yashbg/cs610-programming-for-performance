// Compile: g++ -O2 -fopenmp -o problem2 problem2.cpp

#include <cassert>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <omp.h>

using std::cout;
using std::endl;

#define N (1 << 24)

// Number of array elements a task will process
#define GRANULARITY (1 << 10)

uint64_t reference_sum(uint32_t* A) {
  uint64_t seq_sum = 0;
  for (int i = 0; i < N; i++) {
    seq_sum += A[i];
  }
  return seq_sum;
}

uint64_t par_sum_omp_nored(uint32_t* A) {
  // SB: Write your OpenMP code here
  return 0;
}

uint64_t par_sum_omp_red(uint32_t* A) {
  // SB: Write your OpenMP code here
  return 0;
}

uint64_t par_sum_omp_tasks(uint32_t* A) {
  // SB: Write your OpenMP code here
  return 0;
}

int main() {
  uint32_t* x = new uint32_t[N];
  for (int i = 0; i < N; i++) {
    x[i] = i;
  }

  double start_time, end_time, pi;

  start_time = omp_get_wtime();
  uint64_t seq_sum = reference_sum(x);
  end_time = omp_get_wtime();
  cout << "Sequential sum: " << seq_sum << " in " << (end_time - start_time) << " seconds\n";

  start_time = omp_get_wtime();
  uint64_t par_sum = par_sum_omp_nored(x);
  end_time = omp_get_wtime();
  assert(seq_sum == par_sum);
  cout << "Parallel sum (thread-local, atomic): " << par_sum << " in " << (end_time - start_time)
       << " seconds\n";

  start_time = omp_get_wtime();
  uint64_t ws_sum = par_sum_omp_red(x);
  end_time = omp_get_wtime();
  assert(seq_sum == ws_sum);
  cout << "Parallel sum (worksharing construct): " << ws_sum << " in " << (end_time - start_time)
       << " seconds\n";

  start_time = omp_get_wtime();
  uint64_t task_sum = par_sum_omp_tasks(x);
  end_time = omp_get_wtime();
  if (seq_sum != task_sum) {
    cout << "Seq sum: " << seq_sum << " Task sum: " << task_sum << "\n";
  }
  assert(seq_sum == task_sum);
  cout << "Parallel sum (OpenMP tasks): " << task_sum << " in " << (end_time - start_time)
       << " seconds\n";

  return EXIT_SUCCESS;
}
