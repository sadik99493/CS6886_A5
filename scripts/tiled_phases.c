/*
 * tiled_phases.c -- same code as src/attn_tiled.c, with a timer around
 * each of the three phases (QK^T, softmax, SV). Analysis helper for Q3b (K packing counted in QK^T).
 *
 * Build: $(CC) -O3 -march=native -o build/tiled_phases scripts/tiled_phases.c -lm
 * Usage: build/tiled_phases input/qkv_N4096_d64.bin
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

/*
 * Same three phases as attn_loops.c (S = QK^T*scale, row softmax, O = SV), but
 * both matrix products use
 *   - register tiling: a 4 x 16 output tile is kept in 16 NEON registers
 *     (4 floats each) for the whole reduction, so the inner loop does ~5 loads
 *     per 16 FMAs and no stores; 16 independent accumulators = 4 FMA pipes x
 *     4-cycle FMA latency, enough to keep all FMA units busy.
 *   - cache tiling: the K (resp. V) data used by one block is ~64 KB, so it stays
 *     in the 128 KB L1D while every row block of Q (resp. S) sweeps over it.
 * Softmax is unchanged from attn_loops.c.
 */
#include <arm_neon.h>

#define MR 4                     /* rows of the output tile (rows of Q / S / O)    */
#define NR 16                    /* columns of the output tile = 4 NEON vectors    */
#define L1_BLOCK_FLOATS (16384)  /* 64 KB worth of floats: L1 block budget         */

/* S[i..i+3][j..j+15] = scale * Q[i..i+3][:] . Kp_panel
 * Q    : points at Q[i][0], row stride d
 * Kp   : packed K panel for columns j..j+15, layout [k][c] (d x 16, contiguous)
 * S    : points at S[i][j], row stride ldS                                      */
static void qk_kernel(const float *restrict Q, int d, const float *restrict Kp,
                      float *restrict S, size_t ldS, float scale)
{
    float32x4_t acc[MR][4];
    for (int r = 0; r < MR; r++)
        for (int c = 0; c < 4; c++)
            acc[r][c] = vdupq_n_f32(0.0f);

    for (int k = 0; k < d; k += 4) {
        /* q[r] = Q[i+r][k..k+3]: 4 consecutive k values of row r */
        float32x4_t q[MR];
        for (int r = 0; r < MR; r++)
            q[r] = vld1q_f32(Q + (size_t)r * d + k);

/* one k step: load 16 packed K values, then 16 FMAs.
 * vfmaq_laneq_f32(a, b, q, L) = a + b * q[L]  (q[L] broadcast to all 4 lanes) */
#define QK_STEP(L)                                                                 \
    do {                                                                           \
        const float *bp = Kp + (size_t)(k + L) * NR;                               \
        float32x4_t b0 = vld1q_f32(bp), b1 = vld1q_f32(bp + 4);                    \
        float32x4_t b2 = vld1q_f32(bp + 8), b3 = vld1q_f32(bp + 12);               \
        for (int r = 0; r < MR; r++) {                                             \
            acc[r][0] = vfmaq_laneq_f32(acc[r][0], b0, q[r], L);                   \
            acc[r][1] = vfmaq_laneq_f32(acc[r][1], b1, q[r], L);                   \
            acc[r][2] = vfmaq_laneq_f32(acc[r][2], b2, q[r], L);                   \
            acc[r][3] = vfmaq_laneq_f32(acc[r][3], b3, q[r], L);                   \
        }                                                                          \
    } while (0)
        QK_STEP(0);
        QK_STEP(1);
        QK_STEP(2);
        QK_STEP(3);
#undef QK_STEP
    }

    for (int r = 0; r < MR; r++)
        for (int c = 0; c < 4; c++)
            vst1q_f32(S + (size_t)r * ldS + 4 * c, vmulq_n_f32(acc[r][c], scale));
}

/* O[i..i+3][k0..k0+15] (+)= S[i..i+3][j0..j1) . V[j0..j1)[k0..k0+15]
 * S : points at S[i][0], row stride N;  V: points at V[0][k0], row stride d;
 * O : points at O[i][k0], row stride d
 * first: start from zero instead of the partial sums already in O                */
static void sv_kernel(const float *restrict S, int N, const float *restrict V, int d,
                      float *restrict O, int j0, int j1, int first)
{
    float32x4_t acc[MR][4];
    for (int r = 0; r < MR; r++)
        for (int c = 0; c < 4; c++)
            acc[r][c] = first ? vdupq_n_f32(0.0f) : vld1q_f32(O + (size_t)r * d + 4 * c);

    for (int j = j0; j < j1; j += 4) {
        /* s[r] = S[i+r][j..j+3] */
        float32x4_t s[MR];
        for (int r = 0; r < MR; r++)
            s[r] = vld1q_f32(S + (size_t)r * N + j);

#define SV_STEP(L)                                                                 \
    do {                                                                           \
        const float *vp = V + (size_t)(j + L) * d;                                 \
        float32x4_t v0 = vld1q_f32(vp), v1 = vld1q_f32(vp + 4);                    \
        float32x4_t v2 = vld1q_f32(vp + 8), v3 = vld1q_f32(vp + 12);               \
        for (int r = 0; r < MR; r++) {                                             \
            acc[r][0] = vfmaq_laneq_f32(acc[r][0], v0, s[r], L);                   \
            acc[r][1] = vfmaq_laneq_f32(acc[r][1], v1, s[r], L);                   \
            acc[r][2] = vfmaq_laneq_f32(acc[r][2], v2, s[r], L);                   \
            acc[r][3] = vfmaq_laneq_f32(acc[r][3], v3, s[r], L);                   \
        }                                                                          \
    } while (0)
        SV_STEP(0);
        SV_STEP(1);
        SV_STEP(2);
        SV_STEP(3);
#undef SV_STEP
    }

    for (int r = 0; r < MR; r++)
        for (int c = 0; c < 4; c++)
            vst1q_f32(O + (size_t)r * d + 4 * c, acc[r][c]);
}

static void softmax_rows(float *restrict S, int N)
{
    for (int i = 0; i < N; i++) {
        float *row = S + (size_t)i * N;
        float max = row[0];
        for (int j = 1; j < N; j++)
            max = fmaxf(max, row[j]);
        float sum = 0.0f;
        for (int j = 0; j < N; j++) {
            row[j] = expf(row[j] - max);
            sum += row[j];
        }
        const float inv = 1.0f / sum;
        for (int j = 0; j < N; j++)
            row[j] *= inv;
    }
}

static void attention(const float *restrict Q, const float *restrict K, const float *restrict V,
               float *restrict O, int N, int d, double t[3])
{
    /* The kernels assume tile-aligned sizes; true for every given input. */
    if (N % NR != 0 || d % NR != 0) {
        fprintf(stderr, "attn_tiled: needs N and d to be multiples of %d\n", NR);
        exit(1);
    }
    const float scale = 1.0f / sqrtf((float)d);
    float *restrict S = malloc((size_t)N * N * sizeof(float));
    float *restrict Kp = malloc((size_t)N * d * sizeof(float));

    double t0 = now();
    /* 0. Pack K into panels of NR rows: panel p holds K[p*NR .. p*NR+15][:]
     *    transposed to [k][c], so each k step of qk_kernel reads 16 adjacent floats. */
    for (int p = 0; p < N / NR; p++)
        for (int k = 0; k < d; k++)
            for (int c = 0; c < NR; c++)
                Kp[((size_t)p * d + k) * NR + c] = K[((size_t)p * NR + c) * d + k];

    /* 1. S = Q K^T * scale.
     *    Cache block: JB columns of S, whose packed K data (JB x d floats = 64 KB)
     *    stays in L1 while all N/MR row blocks of Q stream past it. */
    int JB = (L1_BLOCK_FLOATS / d) / NR * NR;
    if (JB < NR)
        JB = NR;
    for (int j0 = 0; j0 < N; j0 += JB) {
        const int j1 = j0 + JB < N ? j0 + JB : N;
        for (int i = 0; i < N; i += MR)
            for (int j = j0; j < j1; j += NR)
                qk_kernel(Q + (size_t)i * d, d, Kp + (size_t)(j / NR) * d * NR,
                          S + (size_t)i * N + j, N, scale);
    }

    double t1 = now();
    /* 2. softmax of each row of S */
    softmax_rows(S, N);
    double t2 = now();

    /* 3. O = S V.
     *    Cache block: JV rows of V (JV x d floats = 64 KB) stay in L1 while all
     *    row blocks of S use them; partial sums are kept in O between blocks. */
    int JV = (L1_BLOCK_FLOATS / d) / 4 * 4;
    if (JV < 4)
        JV = 4;
    for (int j0 = 0; j0 < N; j0 += JV) {
        const int j1 = j0 + JV < N ? j0 + JV : N;
        for (int i = 0; i < N; i += MR)
            for (int k0 = 0; k0 < d; k0 += NR)
                sv_kernel(S + (size_t)i * N, N, V + k0, d, O + (size_t)i * d + k0, j0, j1,
                          j0 == 0);
    }

    double t3 = now();
    t[0] = t1 - t0;
    t[1] = t2 - t1;
    t[2] = t3 - t2;
    free(Kp);
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
