// RUN: if [ %llvmver -ge 12 ]; then %clang++ -O2 %s -S -emit-llvm -o - %loadClangEnzyme -mllvm -enzyme-auto-sparsity=1 | %lli - 1 4; fi

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <vector>

extern double *__enzyme_todense(void *, ...) noexcept;

__attribute__((always_inline))
static double load_zero(int64_t, size_t, size_t, double *) { return 0; }

__attribute__((enzyme_sparse_accumulate))
static void accumulate(size_t index, double value, double *result) {
    result[index] += value;
}

__attribute__((always_inline))
static void store(double value, int64_t offset, size_t row, size_t n,
                  double *result) {
    accumulate(row * n + offset / sizeof(double), value, result);
}

template <int operation>
static size_t affine(size_t index, size_t offset) {
    if (operation == 0) return index + offset;
    if (operation == 1) return index - offset;
    return offset - index;
}

template <int operation>
__attribute__((noinline))
static void sparse(size_t n, size_t a, size_t b, double *result) {
    __builtin_assume(n > 0);
    for (size_t row = 0; row < n; ++row) {
        auto output = __enzyme_todense((void *)load_zero, (void *)store,
                                      row, n, result);
        for (size_t i = 0; i < n; ++i)
            if (affine<operation>(i, a) == row || affine<operation>(i, b) == row)
                output[i] = 1;
    }
}

template <int operation>
static bool check(size_t a, size_t b) {
    const size_t n = 16;
    std::vector<double> result(n * n);
    sparse<operation>(n, a, b, result.data());
    for (size_t row = 0; row < n; ++row)
        for (size_t i = 0; i < n; ++i) {
            double expected = affine<operation>(i, a) == row ||
                              affine<operation>(i, b) == row;
            if (result[row * n + i] != expected) {
                fprintf(stderr, "mode %d, row %zu, index %zu: %g != %g\n",
                        operation, row, i, result[row * n + i], expected);
                return false;
            }
        }
    return true;
}

int main(int, char **argv) {
    size_t a = strtoul(argv[1], nullptr, 10);
    size_t b = strtoul(argv[2], nullptr, 10);
    const size_t offsets[][2] = {{a, b}, {b, a}, {a, a}};
    for (const auto &pair : offsets)
        if (!(check<0>(pair[0], pair[1]) && check<1>(pair[0], pair[1]) &&
              check<2>(pair[0], pair[1])))
            return 1;
    return 0;
}
