// Compile: g++ -O2 -mavx -mavx2 -march=native -o problem1 problem1.cpp

#include <cmath>
#include <iostream>
#include <sys/time.h>
#include <unistd.h>
#include <malloc.h>
#include <immintrin.h>

using std::cout;
using std::endl;
using std::ios;

const int N = (1 << 13);
const int Niter = 10;
const double THRESHOLD = 0.000001;

double rtclock() {
  struct timezone Tzp;
  struct timeval Tp;
  int stat;
  stat = gettimeofday(&Tp, &Tzp);
  if (stat != 0) {
    cout << "Error return from gettimeofday: " << stat << endl;
  }
  return (Tp.tv_sec + Tp.tv_usec * 1.0e-6);
}

void reference(double** A, const double* x, double* y_ref, double* z_ref) {
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      y_ref[j] = y_ref[j] + A[i][j] * x[i];
      z_ref[j] = z_ref[j] + A[j][i] * x[i];
    }
  }
}

void check_result(const double* w_ref, const double* w_opt) {
  double maxdiff = 0.0, this_diff = 0.0;
  int numdiffs = 0;

  for (int i = 0; i < N; i++) {
    this_diff = w_ref[i] - w_opt[i];
    if (fabs(this_diff) > THRESHOLD) {
      numdiffs++;
      if (this_diff > maxdiff)
        maxdiff = this_diff;
    }
  }

  if (numdiffs > 0) {
    cout << numdiffs << " Diffs found over threshold " << THRESHOLD << "; Max Diff = " << maxdiff
         << endl;
  } else {
    cout << "No differences found between base and test versions\n";
  }
}

// Loop interchange
void loop_interchange(double** A, const double* x, double* y_opt, double* z_opt) {
  for (int j = 0; j < N; j++) {
    for (int i = 0; i < N; i++) {
      y_opt[j] = y_opt[j] + A[i][j] * x[i];
      z_opt[j] = z_opt[j] + A[j][i] * x[i];
    }
  }
}

// Fission
void fission(double** A, const double* x, double* y_opt, double* z_opt) {
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      y_opt[j] = y_opt[j] + A[i][j] * x[i];
    }

    for (int j = 0; j < N; j++) {
      z_opt[j] = z_opt[j] + A[j][i] * x[i];
    }
  }
}

// 2 times inner loop unrolling
void inner_loop_unroll2(double** A, const double* x, double* y_opt, double* z_opt) {
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j += 2) {
      y_opt[j] = y_opt[j] + A[i][j] * x[i];
      z_opt[j] = z_opt[j] + A[j][i] * x[i];

      y_opt[j + 1] = y_opt[j + 1] + A[i][j + 1] * x[i];
      z_opt[j + 1] = z_opt[j + 1] + A[j + 1][i] * x[i];
    }
  }
}

// 2 times outer loop unrolling + inner loop jamming
void unroll_jam2(double** A, const double* x, double* y_opt, double* z_opt) {
  for (int i = 0; i < N; i += 2) {
    for (int j = 0; j < N; j++) {
      y_opt[j] = y_opt[j] + A[i][j] * x[i];
      z_opt[j] = z_opt[j] + A[j][i] * x[i];

      y_opt[j] = y_opt[j] + A[i + 1][j] * x[i + 1];
      z_opt[j] = z_opt[j] + A[j][i + 1] * x[i + 1];
    }
  }
}

// 2x2 blocking
void blocking2x2(double** A, const double* x, double* y_opt, double* z_opt) {
  for (int it = 0; it < N; it += 2) {
    for (int jt = 0; jt < N; jt += 2) {
      for (int i = it; i < it + 2; i++) {
        for (int j = jt; j < jt + 2; j++) {
          y_opt[j] = y_opt[j] + A[i][j] * x[i];
          z_opt[j] = z_opt[j] + A[j][i] * x[i];
        }
      }
    }
  }
}

// 4 times outer loop unrolling + inner loop jamming
void unroll_jam4(double** A, const double* x, double* y_opt, double* z_opt) {
  for (int i = 0; i < N; i += 4) {
    for (int j = 0; j < N; j++) {
      y_opt[j] = y_opt[j] + A[i][j] * x[i];
      z_opt[j] = z_opt[j] + A[j][i] * x[i];

      y_opt[j] = y_opt[j] + A[i + 1][j] * x[i + 1];
      z_opt[j] = z_opt[j] + A[j][i + 1] * x[i + 1];

      y_opt[j] = y_opt[j] + A[i + 2][j] * x[i + 2];
      z_opt[j] = z_opt[j] + A[j][i + 2] * x[i + 2];

      y_opt[j] = y_opt[j] + A[i + 3][j] * x[i + 3];
      z_opt[j] = z_opt[j] + A[j][i + 3] * x[i + 3];
    }
  }
}

// 4x4 blocking
void blocking4x4(double** A, const double* x, double* y_opt, double* z_opt) {
  for (int it = 0; it < N; it += 4) {
    for (int jt = 0; jt < N; jt += 4) {
      for (int i = it; i < it + 4; i++) {
        for (int j = jt; j < jt + 4; j++) {
          y_opt[j] = y_opt[j] + A[i][j] * x[i];
          z_opt[j] = z_opt[j] + A[j][i] * x[i];
        }
      }
    }
  }
}

// 8 times outer loop unrolling + inner loop jamming
void unroll_jam8(double** A, const double* x, double* y_opt, double* z_opt) {
  for (int i = 0; i < N; i += 8) {
    for (int j = 0; j < N; j++) {
      y_opt[j] = y_opt[j] + A[i][j] * x[i];
      z_opt[j] = z_opt[j] + A[j][i] * x[i];

      y_opt[j] = y_opt[j] + A[i + 1][j] * x[i + 1];
      z_opt[j] = z_opt[j] + A[j][i + 1] * x[i + 1];

      y_opt[j] = y_opt[j] + A[i + 2][j] * x[i + 2];
      z_opt[j] = z_opt[j] + A[j][i + 2] * x[i + 2];

      y_opt[j] = y_opt[j] + A[i + 3][j] * x[i + 3];
      z_opt[j] = z_opt[j] + A[j][i + 3] * x[i + 3];

      y_opt[j] = y_opt[j] + A[i + 4][j] * x[i + 4];
      z_opt[j] = z_opt[j] + A[j][i + 4] * x[i + 4];

      y_opt[j] = y_opt[j] + A[i + 5][j] * x[i + 5];
      z_opt[j] = z_opt[j] + A[j][i + 5] * x[i + 5];

      y_opt[j] = y_opt[j] + A[i + 6][j] * x[i + 6];
      z_opt[j] = z_opt[j] + A[j][i + 6] * x[i + 6];

      y_opt[j] = y_opt[j] + A[i + 7][j] * x[i + 7];
      z_opt[j] = z_opt[j] + A[j][i + 7] * x[i + 7];
    }
  }
}

// 8x8 blocking
void blocking8x8(double** A, const double* x, double* y_opt, double* z_opt) {
  for (int it = 0; it < N; it += 8) {
    for (int jt = 0; jt < N; jt += 8) {
      for (int i = it; i < it + 8; i++) {
        for (int j = jt; j < jt + 8; j++) {
          y_opt[j] = y_opt[j] + A[i][j] * x[i];
          z_opt[j] = z_opt[j] + A[j][i] * x[i];
        }
      }
    }
  }
}

// 4 times outer loop unrolling + inner loop jamming + ivdep
void unroll_jam4_ivdep(double** A, const double* x, double* y_opt, double* z_opt) {
  for (int i = 0; i < N; i += 4) {
    #pragma GCC ivdep
    for (int j = 0; j < N; j++) {
      y_opt[j] = y_opt[j] + A[i][j] * x[i];
      z_opt[j] = z_opt[j] + A[j][i] * x[i];

      y_opt[j] = y_opt[j] + A[i + 1][j] * x[i + 1];
      z_opt[j] = z_opt[j] + A[j][i + 1] * x[i + 1];

      y_opt[j] = y_opt[j] + A[i + 2][j] * x[i + 2];
      z_opt[j] = z_opt[j] + A[j][i + 2] * x[i + 2];

      y_opt[j] = y_opt[j] + A[i + 3][j] * x[i + 3];
      z_opt[j] = z_opt[j] + A[j][i + 3] * x[i + 3];
    }
  }
}

// 4x4 blocking + ivdep
void blocking4x4_ivdep(double** A, const double* x, double* y_opt, double* z_opt) {
  for (int it = 0; it < N; it += 4) {
    for (int jt = 0; jt < N; jt += 4) {
      for (int i = it; i < it + 4; i++) {
        #pragma GCC ivdep
        for (int j = jt; j < jt + 4; j++) {
          y_opt[j] = y_opt[j] + A[i][j] * x[i];
          z_opt[j] = z_opt[j] + A[j][i] * x[i];
        }
      }
    }
  }
}

// 4 times outer loop unrolling + inner loop jamming + ivdep + restrict
void unroll_jam4_ivdep_restrict(double** __restrict__ A, const double* __restrict__ x, double* __restrict__ y_opt, double* __restrict__ z_opt) {
  for (int i = 0; i < N; i += 4) {
    #pragma GCC ivdep
    for (int j = 0; j < N; j++) {
      y_opt[j] = y_opt[j] + A[i][j] * x[i];
      z_opt[j] = z_opt[j] + A[j][i] * x[i];

      y_opt[j] = y_opt[j] + A[i + 1][j] * x[i + 1];
      z_opt[j] = z_opt[j] + A[j][i + 1] * x[i + 1];

      y_opt[j] = y_opt[j] + A[i + 2][j] * x[i + 2];
      z_opt[j] = z_opt[j] + A[j][i + 2] * x[i + 2];

      y_opt[j] = y_opt[j] + A[i + 3][j] * x[i + 3];
      z_opt[j] = z_opt[j] + A[j][i + 3] * x[i + 3];
    }
  }
}

// 4x4 blocking + ivdep + restrict
void blocking4x4_ivdep_restrict(double** __restrict__ A, const double* __restrict__ x, double* __restrict__ y_opt, double* __restrict__ z_opt) {
  for (int it = 0; it < N; it += 4) {
    for (int jt = 0; jt < N; jt += 4) {
      for (int i = it; i < it + 4; i++) {
        #pragma GCC ivdep
        for (int j = jt; j < jt + 4; j++) {
          y_opt[j] = y_opt[j] + A[i][j] * x[i];
          z_opt[j] = z_opt[j] + A[j][i] * x[i];
        }
      }
    }
  }
}

// 4 times outer loop unrolling + inner loop jamming + ivdep + restrict + aligned
void unroll_jam4_ivdep_restrict_aligned(double** __restrict__ A, const double* __restrict__ x, double* __restrict__ y_opt, double* __restrict__ z_opt) {
  A = (double**)__builtin_assume_aligned(A, 32);
  x = (double*)__builtin_assume_aligned(x, 32);
  y_opt = (double*)__builtin_assume_aligned(y_opt, 32);
  z_opt = (double*)__builtin_assume_aligned(z_opt, 32);

  for (int i = 0; i < N; i += 4) {
    #pragma GCC ivdep
    for (int j = 0; j < N; j++) {
      y_opt[j] = y_opt[j] + A[i][j] * x[i];
      z_opt[j] = z_opt[j] + A[j][i] * x[i];

      y_opt[j] = y_opt[j] + A[i + 1][j] * x[i + 1];
      z_opt[j] = z_opt[j] + A[j][i + 1] * x[i + 1];

      y_opt[j] = y_opt[j] + A[i + 2][j] * x[i + 2];
      z_opt[j] = z_opt[j] + A[j][i + 2] * x[i + 2];

      y_opt[j] = y_opt[j] + A[i + 3][j] * x[i + 3];
      z_opt[j] = z_opt[j] + A[j][i + 3] * x[i + 3];
    }
  }
}

// 4x4 blocking + ivdep + restrict + aligned
void blocking4x4_ivdep_restrict_aligned(double** __restrict__ A, const double* __restrict__ x, double* __restrict__ y_opt, double* __restrict__ z_opt) {
  A = (double**)__builtin_assume_aligned(A, 32);
  x = (double*)__builtin_assume_aligned(x, 32);
  y_opt = (double*)__builtin_assume_aligned(y_opt, 32);
  z_opt = (double*)__builtin_assume_aligned(z_opt, 32);

  for (int it = 0; it < N; it += 4) {
    for (int jt = 0; jt < N; jt += 4) {
      for (int i = it; i < it + 4; i++) {
        #pragma GCC ivdep
        for (int j = jt; j < jt + 4; j++) {
          y_opt[j] = y_opt[j] + A[i][j] * x[i];
          z_opt[j] = z_opt[j] + A[j][i] * x[i];
        }
      }
    }
  }
}

// Intrinsics Version: 4 times outer loop unrolling + inner loop jamming + ivdep + restrict + aligned
void avx_version(double** A, double* x, double* y_opt, double* z_opt) {
  __m256d rA_ij, rA_ji, rx_i, ry_j, rz_j;
  for (int i = 0; i < N; i += 4) {
    for (int j = 0; j < N; j += 4) {
      // y_opt[j] = y_opt[j] + A[i][j] * x[i];
      // z_opt[j] = z_opt[j] + A[j][i] * x[i];

      rA_ij = _mm256_load_pd(&A[i][j]);
      rA_ji = _mm256_set_pd(A[j + 3][i], A[j + 2][i], A[j + 1][i], A[j][i]);
      rx_i = _mm256_set1_pd(x[i]);
      ry_j = _mm256_load_pd(&y_opt[j]);
      rz_j = _mm256_load_pd(&z_opt[j]);

      ry_j = _mm256_add_pd(ry_j, _mm256_mul_pd(rA_ij, rx_i));
      rz_j = _mm256_add_pd(rz_j, _mm256_mul_pd(rA_ji, rx_i));

      _mm256_store_pd(&y_opt[j], ry_j);
      _mm256_store_pd(&z_opt[j], rz_j);

      // y_opt[j] = y_opt[j] + A[i + 1][j] * x[i + 1];
      // z_opt[j] = z_opt[j] + A[j][i + 1] * x[i + 1];

      rA_ij = _mm256_load_pd(&A[i + 1][j]);
      rA_ji = _mm256_set_pd(A[j + 3][i + 1], A[j + 2][i + 1], A[j + 1][i + 1], A[j][i + 1]);
      rx_i = _mm256_set1_pd(x[i + 1]);
      ry_j = _mm256_load_pd(&y_opt[j]);
      rz_j = _mm256_load_pd(&z_opt[j]);

      ry_j = _mm256_add_pd(ry_j, _mm256_mul_pd(rA_ij, rx_i));
      rz_j = _mm256_add_pd(rz_j, _mm256_mul_pd(rA_ji, rx_i));

      _mm256_store_pd(&y_opt[j], ry_j);
      _mm256_store_pd(&z_opt[j], rz_j);

      // y_opt[j] = y_opt[j] + A[i + 2][j] * x[i + 2];
      // z_opt[j] = z_opt[j] + A[j][i + 2] * x[i + 2];

      rA_ij = _mm256_load_pd(&A[i + 2][j]);
      rA_ji = _mm256_set_pd(A[j + 3][i + 2], A[j + 2][i + 2], A[j + 1][i + 2], A[j][i + 2]);
      rx_i = _mm256_set1_pd(x[i + 2]);
      ry_j = _mm256_load_pd(&y_opt[j]);
      rz_j = _mm256_load_pd(&z_opt[j]);

      ry_j = _mm256_add_pd(ry_j, _mm256_mul_pd(rA_ij, rx_i));
      rz_j = _mm256_add_pd(rz_j, _mm256_mul_pd(rA_ji, rx_i));

      _mm256_store_pd(&y_opt[j], ry_j);
      _mm256_store_pd(&z_opt[j], rz_j);

      // y_opt[j] = y_opt[j] + A[i + 3][j] * x[i + 3];
      // z_opt[j] = z_opt[j] + A[j][i + 3] * x[i + 3];

      rA_ij = _mm256_load_pd(&A[i + 3][j]);
      rA_ji = _mm256_set_pd(A[j + 3][i + 3], A[j + 2][i + 3], A[j + 1][i + 3], A[j][i + 3]);
      rx_i = _mm256_set1_pd(x[i + 3]);
      ry_j = _mm256_load_pd(&y_opt[j]);
      rz_j = _mm256_load_pd(&z_opt[j]);

      ry_j = _mm256_add_pd(ry_j, _mm256_mul_pd(rA_ij, rx_i));
      rz_j = _mm256_add_pd(rz_j, _mm256_mul_pd(rA_ji, rx_i));

      _mm256_store_pd(&y_opt[j], ry_j);
      _mm256_store_pd(&z_opt[j], rz_j);
    }
  }
}

int main() {
  double clkbegin, clkend;
  double t;
  double reftime, opttime;

  cout.setf(ios::fixed, ios::floatfield);
  cout.precision(5);

  double** A;
  A = new double*[N];
  for (int i = 0; i < N; i++) {
    A[i] = new double[N];
  }

  double *x, *y_ref, *z_ref, *y_opt, *z_opt;
  x = new double[N];
  y_ref = new double[N];
  z_ref = new double[N];
  y_opt = new double[N];
  z_opt = new double[N];

  for (int i = 0; i < N; i++) {
    x[i] = i;
    y_ref[i] = 1.0;
    y_opt[i] = 1.0;
    z_ref[i] = 2.0;
    z_opt[i] = 2.0;
    for (int j = 0; j < N; j++) {
      A[i][j] = (i + 2.0 * j) / (2.0 * N);
    }
  }

  // Reference version
  clkbegin = rtclock();
  for (int it = 0; it < Niter; it++) {
    reference(A, x, y_ref, z_ref);
  }
  clkend = rtclock();
  t = clkend - clkbegin;
  reftime = t / Niter;
  cout << "Reference Version: Matrix Size = " << N << ", " << 4.0 * 1e-9 * N * N * Niter / t
       << " GFLOPS; Time = " << t / Niter << " sec" << endl << endl;

  // Loop interchange
  clkbegin = rtclock();
  for (int it = 0; it < Niter; it++) {
    loop_interchange(A, x, y_opt, z_opt);
  }
  clkend = rtclock();
  t = clkend - clkbegin;
  opttime = t / Niter;
  cout << "Loop interchange: Matrix Size = " << N << ", Time = " << t / Niter << " sec, Speedup = " << reftime / opttime << endl;
  check_result(y_ref, y_opt);
  cout << endl;

  // Reset
  for (int i = 0; i < N; i++) {
    y_opt[i] = 1.0;
    z_opt[i] = 2.0;
  }

  // Fission
  clkbegin = rtclock();
  for (int it = 0; it < Niter; it++) {
    fission(A, x, y_opt, z_opt);
  }
  clkend = rtclock();
  t = clkend - clkbegin;
  opttime = t / Niter;
  cout << "Fission: Matrix Size = " << N << ", Time = " << t / Niter << " sec, Speedup = " << reftime / opttime << endl;
  check_result(y_ref, y_opt);
  cout << endl;

  // Reset
  for (int i = 0; i < N; i++) {
    y_opt[i] = 1.0;
    z_opt[i] = 2.0;
  }

  // 2 times inner loop unrolling
  clkbegin = rtclock();
  for (int it = 0; it < Niter; it++) {
    inner_loop_unroll2(A, x, y_opt, z_opt);
  }
  clkend = rtclock();
  t = clkend - clkbegin;
  opttime = t / Niter;
  cout << "2 times inner loop unrolling: Matrix Size = " << N << ", Time = " << t / Niter << " sec, Speedup = " << reftime / opttime << endl;
  check_result(y_ref, y_opt);
  cout << endl;

  // Reset
  for (int i = 0; i < N; i++) {
    y_opt[i] = 1.0;
    z_opt[i] = 2.0;
  }

  // 2 times outer loop unrolling + inner loop jamming
  clkbegin = rtclock();
  for (int it = 0; it < Niter; it++) {
    unroll_jam2(A, x, y_opt, z_opt);
  }
  clkend = rtclock();
  t = clkend - clkbegin;
  opttime = t / Niter;
  cout << "2 times outer loop unrolling + inner loop jamming: Matrix Size = " << N << ", Time = " << t / Niter << " sec, Speedup = " << reftime / opttime << endl;
  check_result(y_ref, y_opt);
  cout << endl;

  // Reset
  for (int i = 0; i < N; i++) {
    y_opt[i] = 1.0;
    z_opt[i] = 2.0;
  }

  // 2x2 blocking
  clkbegin = rtclock();
  for (int it = 0; it < Niter; it++) {
    blocking2x2(A, x, y_opt, z_opt);
  }
  clkend = rtclock();
  t = clkend - clkbegin;
  opttime = t / Niter;
  cout << "2x2 blocking: Matrix Size = " << N << ", Time = " << t / Niter << " sec, Speedup = " << reftime / opttime << endl;
  check_result(y_ref, y_opt);
  cout << endl;

  // Reset
  for (int i = 0; i < N; i++) {
    y_opt[i] = 1.0;
    z_opt[i] = 2.0;
  }

  // 4 times outer loop unrolling + inner loop jamming
  clkbegin = rtclock();
  for (int it = 0; it < Niter; it++) {
    unroll_jam4(A, x, y_opt, z_opt);
  }
  clkend = rtclock();
  t = clkend - clkbegin;
  opttime = t / Niter;
  cout << "4 times outer loop unrolling + inner loop jamming: Matrix Size = " << N << ", Time = " << t / Niter << " sec, Speedup = " << reftime / opttime << endl;
  check_result(y_ref, y_opt);
  cout << endl;

  // Reset
  for (int i = 0; i < N; i++) {
    y_opt[i] = 1.0;
    z_opt[i] = 2.0;
  }

  // 4x4 blocking
  clkbegin = rtclock();
  for (int it = 0; it < Niter; it++) {
    blocking4x4(A, x, y_opt, z_opt);
  }
  clkend = rtclock();
  t = clkend - clkbegin;
  opttime = t / Niter;
  cout << "4x4 blocking: Matrix Size = " << N << ", Time = " << t / Niter << " sec, Speedup = " << reftime / opttime << endl;
  check_result(y_ref, y_opt);
  cout << endl;

  // Reset
  for (int i = 0; i < N; i++) {
    y_opt[i] = 1.0;
    z_opt[i] = 2.0;
  }

  // 8 times outer loop unrolling + inner loop jamming
  clkbegin = rtclock();
  for (int it = 0; it < Niter; it++) {
    unroll_jam8(A, x, y_opt, z_opt);
  }
  clkend = rtclock();
  t = clkend - clkbegin;
  opttime = t / Niter;
  cout << "8 times outer loop unrolling + inner loop jamming: Matrix Size = " << N << ", Time = " << t / Niter << " sec, Speedup = " << reftime / opttime << endl;
  check_result(y_ref, y_opt);
  cout << endl;

  // Reset
  for (int i = 0; i < N; i++) {
    y_opt[i] = 1.0;
    z_opt[i] = 2.0;
  }

  // 8x8 blocking
  clkbegin = rtclock();
  for (int it = 0; it < Niter; it++) {
    blocking8x8(A, x, y_opt, z_opt);
  }
  clkend = rtclock();
  t = clkend - clkbegin;
  opttime = t / Niter;
  cout << "8x8 blocking: Matrix Size = " << N << ", Time = " << t / Niter << " sec, Speedup = " << reftime / opttime << endl;
  check_result(y_ref, y_opt);
  cout << endl;

  // Reset
  for (int i = 0; i < N; i++) {
    y_opt[i] = 1.0;
    z_opt[i] = 2.0;
  }

  // 4 times outer loop unrolling + inner loop jamming + ivdep
  clkbegin = rtclock();
  for (int it = 0; it < Niter; it++) {
    unroll_jam4_ivdep(A, x, y_opt, z_opt);
  }
  clkend = rtclock();
  t = clkend - clkbegin;
  opttime = t / Niter;
  cout << "4 times outer loop unrolling + inner loop jamming + ivdep: Matrix Size = " << N << ", Time = " << t / Niter << " sec, Speedup = " << reftime / opttime << endl;
  check_result(y_ref, y_opt);
  cout << endl;

  // Reset
  for (int i = 0; i < N; i++) {
    y_opt[i] = 1.0;
    z_opt[i] = 2.0;
  }

  // 4x4 blocking + ivdep
  clkbegin = rtclock();
  for (int it = 0; it < Niter; it++) {
    blocking4x4_ivdep(A, x, y_opt, z_opt);
  }
  clkend = rtclock();
  t = clkend - clkbegin;
  opttime = t / Niter;
  cout << "4x4 blocking + ivdep: Matrix Size = " << N << ", Time = " << t / Niter << " sec, Speedup = " << reftime / opttime << endl;
  check_result(y_ref, y_opt);
  cout << endl;

  // Reset
  for (int i = 0; i < N; i++) {
    y_opt[i] = 1.0;
    z_opt[i] = 2.0;
  }

  // 4 times outer loop unrolling + inner loop jamming + ivdep + restrict
  clkbegin = rtclock();
  for (int it = 0; it < Niter; it++) {
    unroll_jam4_ivdep_restrict(A, x, y_opt, z_opt);
  }
  clkend = rtclock();
  t = clkend - clkbegin;
  opttime = t / Niter;
  cout << "4 times outer loop unrolling + inner loop jamming + ivdep + restrict: Matrix Size = " << N << ", Time = " << t / Niter << " sec, Speedup = " << reftime / opttime << endl;
  check_result(y_ref, y_opt);
  cout << endl;

  // Reset
  for (int i = 0; i < N; i++) {
    y_opt[i] = 1.0;
    z_opt[i] = 2.0;
  }

  // 4x4 blocking + ivdep + restrict
  clkbegin = rtclock();
  for (int it = 0; it < Niter; it++) {
    blocking4x4_ivdep_restrict(A, x, y_opt, z_opt);
  }
  clkend = rtclock();
  t = clkend - clkbegin;
  opttime = t / Niter;
  cout << "4x4 blocking + ivdep + restrict: Matrix Size = " << N << ", Time = " << t / Niter << " sec, Speedup = " << reftime / opttime << endl;
  check_result(y_ref, y_opt);
  cout << endl;

  // Reset
  for (int i = 0; i < N; i++) {
    y_opt[i] = 1.0;
    z_opt[i] = 2.0;
  }

  for (int i = 0; i < N; i++) {
    delete[] A[i];
  }
  delete[] A;
  
  delete[] x;
  delete[] y_opt;
  delete[] z_opt;

  // Aligned versions

  A = (double**)memalign(32, N * sizeof(double));
  for (int i = 0; i < N; i++) {
    A[i] = (double*)memalign(32, N * sizeof(double));
  }

  x = (double*)memalign(32, N * sizeof(double));
  y_opt = (double*)memalign(32, N * sizeof(double));
  z_opt = (double*)memalign(32, N * sizeof(double));

  for (int i = 0; i < N; i++) {
    x[i] = i;
    y_opt[i] = 1.0;
    z_opt[i] = 2.0;
    for (int j = 0; j < N; j++) {
      A[i][j] = (i + 2.0 * j) / (2.0 * N);
    }
  }

  // 4 times outer loop unrolling + inner loop jamming + ivdep + restrict + aligned
  clkbegin = rtclock();
  for (int it = 0; it < Niter; it++) {
    unroll_jam4_ivdep_restrict_aligned(A, x, y_opt, z_opt);
  }
  clkend = rtclock();
  t = clkend - clkbegin;
  opttime = t / Niter;
  cout << "4 times outer loop unrolling + inner loop jamming + ivdep + restrict + aligned: Matrix Size = " << N << ", Time = " << t / Niter << " sec, Speedup = " << reftime / opttime << endl;
  check_result(y_ref, y_opt);
  cout << endl;

  // Reset
  for (int i = 0; i < N; i++) {
    y_opt[i] = 1.0;
    z_opt[i] = 2.0;
  }

  // 4x4 blocking + ivdep + restrict + aligned
  clkbegin = rtclock();
  for (int it = 0; it < Niter; it++) {
    blocking4x4_ivdep_restrict_aligned(A, x, y_opt, z_opt);
  }
  clkend = rtclock();
  t = clkend - clkbegin;
  opttime = t / Niter;
  cout << "4x4 blocking + ivdep + restrict + aligned: Matrix Size = " << N << ", Time = " << t / Niter << " sec, Speedup = " << reftime / opttime << endl;
  check_result(y_ref, y_opt);
  cout << endl;

  // Reset
  for (int i = 0; i < N; i++) {
    y_opt[i] = 1.0;
    z_opt[i] = 2.0;
  }

  // Version with intinsics: 4 times outer loop unrolling + inner loop jamming + ivdep + restrict + aligned

  clkbegin = rtclock();
  for (int it = 0; it < Niter; it++) {
    avx_version(A, x, y_opt, z_opt);
  }
  clkend = rtclock();
  t = clkend - clkbegin;
  opttime = t / Niter;
  cout << "Intrinsics Version: 4 times outer loop unrolling + inner loop jamming + ivdep + restrict + aligned: Matrix Size = " << N << ", Time = " << t / Niter << " sec, Speedup = " << reftime / opttime << endl;
  check_result(y_ref, y_opt);

  return EXIT_SUCCESS;
}
