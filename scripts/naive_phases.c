/*
 * naive_phases.c -- same algorithm as src/attn_naive.c, with a timer around
 * each of the three phases (QK^T, softmax, SV). Analysis helper for Q2 only.
 *
 * Build: $(CC) -O3 -march=native -o build/naive_phases scripts/naive_phases.c -lm
 * Usage: build/naive_phases input/qkv_N4096_d64.bin
 * Prints the median (of 5 runs, after 1 warm-up) time per phase.
 */
#define _POSIX_C_SOURCE 200809L
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define REPS 5

static double now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static int cmp_double(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

/* t[0..2] receive the time of each phase */
static void attention(const float *Q, const float *K, const float *V, float *O, int N, int d,
                      double t[3])
{
    const float scale = 1.0f / sqrtf((float)d);
    float *S = malloc((size_t)N * N * sizeof(float));

    double t0 = now();
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            float acc = 0.0f;
            for (int k = 0; k < d; k++)
                acc += Q[(size_t)i * d + k] * K[(size_t)j * d + k];
            S[(size_t)i * N + j] = acc * scale;
        }
    double t1 = now();

    for (int i = 0; i < N; i++) {
        float *row = S + (size_t)i * N;
        float max = row[0];
        for (int j = 1; j < N; j++)
            if (row[j] > max)
                max = row[j];
        float sum = 0.0f;
        for (int j = 0; j < N; j++) {
            row[j] = expf(row[j] - max);
            sum += row[j];
        }
        for (int j = 0; j < N; j++)
            row[j] = row[j] / sum;
    }
    double t2 = now();

    for (int i = 0; i < N; i++)
        for (int k = 0; k < d; k++) {
            float acc = 0.0f;
            for (int j = 0; j < N; j++)
                acc += S[(size_t)i * N + j] * V[(size_t)j * d + k];
            O[(size_t)i * d + k] = acc;
        }
    double t3 = now();

    t[0] = t1 - t0;
    t[1] = t2 - t1;
    t[2] = t3 - t2;
    free(S);
}

int main(int argc, char **argv)
{
    if (argc != 2) {
        fprintf(stderr, "usage: %s <input.bin>\n", argv[0]);
        return 1;
    }
    FILE *in = fopen(argv[1], "rb");
    int32_t hdr[2];
    if (!in || fread(hdr, sizeof hdr, 1, in) != 1) {
        fprintf(stderr, "%s: cannot read\n", argv[1]);
        return 1;
    }
    int N = hdr[0], d = hdr[1];
    size_t n = (size_t)N * d;
    float *Q = malloc(n * 4), *K = malloc(n * 4), *V = malloc(n * 4), *O = malloc(n * 4);
    if (fread(Q, 4, n, in) != n || fread(K, 4, n, in) != n || fread(V, 4, n, in) != n) {
        fprintf(stderr, "%s: short read\n", argv[1]);
        return 1;
    }
    fclose(in);

    double t[3], ph[3][REPS];
    attention(Q, K, V, O, N, d, t); /* warm-up */
    for (int r = 0; r < REPS; r++) {
        attention(Q, K, V, O, N, d, t);
        for (int p = 0; p < 3; p++)
            ph[p][r] = t[p];
    }
    const char *name[3] = {"QK^T", "softmax", "SV"};
    double med[3], total = 0;
    for (int p = 0; p < 3; p++) {
        qsort(ph[p], REPS, sizeof(double), cmp_double);
        med[p] = ph[p][REPS / 2];
        total += med[p];
    }
    printf("N=%d d=%d\n", N, d);
    for (int p = 0; p < 3; p++) {
        double flops = (p == 1) ? 0 : 2.0 * N * (double)N * d;
        printf("  %-8s %9.2f ms  (%4.1f%%)", name[p], med[p] * 1e3, 100 * med[p] / total);
        if (flops > 0)
            printf("  %6.2f GFLOP/s", flops / med[p] * 1e-9);
        printf("\n");
    }
    printf("  total    %9.2f ms\n", total * 1e3);
    return 0;
}
