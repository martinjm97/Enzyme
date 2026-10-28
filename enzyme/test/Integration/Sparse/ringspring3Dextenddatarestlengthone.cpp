// This should work on LLVM 7, 8, 9, however in CI the version of clang installed on Ubuntu 18.04 cannot load
// a clang plugin properly without segfaulting on exit. This is fine on Ubuntu 20.04 or later LLVM versions...
// This should work on LLVM 7, 8, 9, however in CI the version of clang installed on Ubuntu 18.04 cannot load
// a clang plugin properly without segfaulting on exit. This is fine on Ubuntu 20.04 or later LLVM versions...
// RUN: if [ %llvmver -ge 12 ]; then %clang++ -fno-exceptions  -ffast-math -mllvm -enable-load-pre=0 -std=c++11 -O1 %s -S -emit-llvm -o - %loadClangEnzyme -mllvm -enzyme-auto-sparsity=1 | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang++ -fno-exceptions  -ffast-math -mllvm -enable-load-pre=0 -std=c++11 -O2 %s -S -emit-llvm -o - %loadClangEnzyme -mllvm -enzyme-auto-sparsity=1  | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang++ -fno-exceptions  -ffast-math -mllvm -enable-load-pre=0 -std=c++11 -O3 %s -S -emit-llvm -o - %loadClangEnzyme  -mllvm -enzyme-auto-sparsity=1 | %lli - ; fi
// TODO: if [ %llvmver -ge 12 ]; then %clang++ -fno-exceptions -ffast-math -mllvm -enable-load-pre=0  -std=c++11 -O1 %s -S -emit-llvm -o - %newLoadClangEnzyme -mllvm -enzyme-auto-sparsity=1 -S | %lli - ; fi
// TODO: if [ %llvmver -ge 12 ]; then %clang++ -fno-exceptions -ffast-math -mllvm -enable-load-pre=0  -std=c++11 -O2 %s -S -emit-llvm -o - %newLoadClangEnzyme -mllvm -enzyme-auto-sparsity=1 -S | %lli - ; fi
// TODO: if [ %llvmver -ge 12 ]; then %clang++ -fno-exceptions -ffast-math -mllvm -enable-load-pre=0  -std=c++11 -O3 %s -S -emit-llvm -o - %newLoadClangEnzyme -mllvm -enzyme-auto-sparsity=1 -S | %lli - ; fi

// everything should be always inline

#include <stdio.h>
#include <assert.h>
#include <vector>
#include <random>


#include<math.h>

struct triple {
    size_t row;
    size_t col;
    double val;
    triple(triple&&) = default;
    triple(size_t row, size_t col, double val) : row(row), col(col), val(val) {}
};


extern int enzyme_dup;
extern int enzyme_dupnoneed;
extern int enzyme_out;
extern int enzyme_const;

extern void __enzyme_autodiff(void *, ...);

extern void __enzyme_fwddiff(void *, ...);

extern double* __enzyme_todense(void *, ...) noexcept;


__attribute__((always_inline))
static double f(size_t N, double* pos) {
    double e = 0.;
    // A vertex count gives the reverse loop a closed-form trip count.
    for (size_t vertex = 0; vertex < N / 3; ++vertex) {
        size_t i = 3 * vertex;
        double vx = pos[i];
        double vy = pos[i + 1];
        double vz = pos[i + 2];

        double wx = pos[i + 3];
        double wy = pos[i + 4];
        double wz = pos[i + 5];
        double distance = (wx - vx) * (wx - vx) + (wy - vy) * (wy - vy) + (wz - vz) * (wz - vz);
        double rest_len_one_dist = (sqrt(distance) - 1) * (sqrt(distance) - 1);
        e += rest_len_one_dist;
    }
    return e;
}


__attribute__((always_inline))
static void grad_f(size_t N, double* input, double* dinput) {
    __enzyme_autodiff((void*)f, enzyme_const, N, enzyme_dup, input, dinput);
}


__attribute__((always_inline))
static void ident_store(double , int64_t idx, size_t i) {
    assert(0 && "should never load");
}

__attribute__((always_inline))
static double ident_load(int64_t idx, size_t i, size_t N) {
    idx /= sizeof(double);
    return (double)(idx == i);// ? 1.0 : 0.0;
}

__attribute__((enzyme_sparse_accumulate))
static void inner_store(int64_t row, int64_t col, size_t N, double val, std::vector<triple> &triplets) {
    // assert(abs(val) > 0.00001);
    triplets.emplace_back(row % N, col % N, val);
}

__attribute__((always_inline))
static void sparse_store(double val, int64_t idx, size_t i, size_t N, std::vector<triple> &triplets) {
    if (val == 0.0) return;
    idx /= sizeof(double);
    inner_store(i, idx, N, val, triplets);
}

__attribute__((always_inline))
static double sparse_load(int64_t idx, size_t i, size_t N, std::vector<triple> &triplets) {
    return 0.0;
}

__attribute__((always_inline))
static void never_store(double val, int64_t idx, double* input, size_t N) {
    assert(0 && "this is a read only input, why are you storing here...");
}

__attribute__((always_inline))
static double mod_load(int64_t idx, double* input, size_t N) {
    idx /= sizeof(double);
    return input[idx % N];
}

__attribute__((noinline))
std::vector<triple> hess_f(size_t N, double* input) {
    std::vector<triple> triplets;
    // input = __enzyme_todense((void*)mod_load, (void*)never_store, input, N);
    __builtin_assume(N > 0);
    // Include the duplicated closing vertex; inner_store folds periodic indices.
    for (size_t i=0; i<N+3; i++) {
        __builtin_assume(i < 100000000);
        double* d_input = __enzyme_todense((void*)ident_load, (void*)ident_store, i, N);
        double* d_dinput = __enzyme_todense((void*)sparse_load, (void*)sparse_store, i, N, &triplets);

       __enzyme_fwddiff((void*)grad_f, 
                            enzyme_const, N,
                            enzyme_dup, input, d_input,
                            enzyme_dupnoneed, (double*)0x1, d_dinput);

    }
    return triplets;
}

/*
__attribute__((noinline))
std::vector<triple> hess_f2(size_t N, double* input) {
    std::vector<triple> triplets;
    input = 
    ((void*)mod_load, (void*)never_store, input, N);
    hess_f(N, input);
}
*/

// int argc, char** argv
int main() {
    for (size_t vertices : {3, 8, 17}) {
        const size_t N = 3 * vertices;
        std::vector<double> x(N + 3);
        for (size_t i = 0; i < vertices; ++i) {
            double angle = 2 * M_PI * i / vertices;
            x[3 * i] = cos(angle);
            x[3 * i + 1] = sin(angle);
            x[3 * i + 2] = 0.1 * cos(2 * angle);
        }
        for (size_t i = 0; i < 3; ++i)
            x[N + i] = x[i];

        // Fold the padded periodic coordinates into both Hessian indices.
        std::vector<double> sparse(N * N), dense(N * N);
        for (const auto &entry : hess_f(N, x.data()))
            sparse[entry.row * N + entry.col] += entry.val;

        // Differentiate the same gradient using actual dense buffers. This
        // bypasses virtual arrays and the sparse-loop optimization under test.
        for (size_t seed = 0; seed < N + 3; ++seed) {
            std::vector<double> direction(N + 3), gradient(N + 3), column(N + 3);
            direction[seed] = 1;
            __enzyme_fwddiff((void *)grad_f, enzyme_const, N,
                             enzyme_dup, x.data(), direction.data(),
                             enzyme_dupnoneed, gradient.data(), column.data());
            for (size_t j = 0; j < N + 3; ++j)
                dense[(seed % N) * N + j % N] += column[j];
        }
        for (size_t i = 0; i < N * N; ++i)
            if (!std::isfinite(sparse[i]) || fabs(sparse[i] - dense[i]) > 1e-11) {
                fprintf(stderr, "Hessian mismatch at %zu: %.17g != %.17g\n",
                        i, sparse[i], dense[i]);
                return 1;
            }
    }
    return 0;
}
