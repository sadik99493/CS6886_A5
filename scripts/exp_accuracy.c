/*
 * exp_accuracy.c -- measure the accuracy and speed of exp_neon (copied verbatim
 * from src/attn_best.c) against libm. Analysis helper for Q3d.
 *
 * Build: $(CC) -O3 -march=native -o build/exp_accuracy scripts/exp_accuracy.c -lm
 * Usage: build/exp_accuracy
 */
#define _POSIX_C_SOURCE 200809L
#include <arm_neon.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

/* exp(x) for 4 floats at once (Cephes-style expf, rel. error ~1e-7 on [-87, 0]).
 * Idea: exp(x) = 2^n * exp(r), with n = round(x / ln2) and r = x - n*ln2 in [-ln2/2, ln2/2].
 *   - exp(r) on that small interval is a degree-7 polynomial (Horner form, FMAs)
 *   - 2^n is built directly in the float exponent bits: (n + 127) << 23
 * Inputs here are x - max <= 0; x is clamped at -87 so 2^n stays a normal float
 * (exp(-87) ~ 1.6e-38, i.e. 0 for softmax purposes).                              */
static inline float32x4_t exp_neon(float32x4_t x)
{
    x = vmaxq_f32(x, vdupq_n_f32(-87.0f));
    /* n = round(x * log2(e)) */
    const float32x4_t n = vrndnq_f32(vmulq_n_f32(x, 1.44269504088896341f));
    /* r = x - n*ln2, with ln2 split in two parts for extra precision (Cody-Waite) */
    float32x4_t r = vfmsq_n_f32(x, n, 0.693359375f);
    r = vfmsq_n_f32(r, n, -2.12194440e-4f);
    /* p(r) ~ exp(r) = 1 + r + r^2 * (c0 + c1 r + c2 r^2 + c3 r^3 + c4 r^4 + c5 r^5) */
    float32x4_t p = vdupq_n_f32(1.9875691500e-4f);
    p = vfmaq_f32(vdupq_n_f32(1.3981999507e-3f), p, r);
    p = vfmaq_f32(vdupq_n_f32(8.3334519073e-3f), p, r);
    p = vfmaq_f32(vdupq_n_f32(4.1665795894e-2f), p, r);
    p = vfmaq_f32(vdupq_n_f32(1.6666665459e-1f), p, r);
    p = vfmaq_f32(vdupq_n_f32(5.0000001201e-1f), p, r);
    const float32x4_t r2 = vmulq_f32(r, r);
    p = vfmaq_f32(vaddq_f32(r, vdupq_n_f32(1.0f)), p, r2);
    /* 2^n via the exponent field */
    const int32x4_t e = vshlq_n_s32(vaddq_s32(vcvtq_s32_f32(n), vdupq_n_s32(127)), 23);
    return vmulq_f32(p, vreinterpretq_f32_s32(e));
}

int main(void)
{
    /* 1. accuracy: 16M evenly spaced points on [-87, 0], vs double-precision exp */
    const int M = 1 << 24;
    float *x = malloc(M * sizeof(float)), *y = malloc(M * sizeof(float));
    for (int i = 0; i < M; i++)
        x[i] = -87.0f * (float)i / (float)(M - 1);
    for (int i = 0; i < M; i += 4)
        vst1q_f32(y + i, exp_neon(vld1q_f32(x + i)));
    double max_rel_neon = 0, max_rel_libm = 0;
    for (int i = 0; i < M; i++) {
        double ref = exp((double)x[i]);
        double en = fabs(y[i] - ref) / ref, el = fabs(expf(x[i]) - ref) / ref;
        if (en > max_rel_neon) max_rel_neon = en;
        if (el > max_rel_libm) max_rel_libm = el;
    }
    printf("max relative error on [-87, 0]:  exp_neon %.2e   libm expf %.2e\n", max_rel_neon,
           max_rel_libm);

    /* 2. speed: time per element, best of 5, same 16M inputs */
    double tn = 1e9, tl = 1e9;
    for (int rep = 0; rep < 5; rep++) {
        double t0 = now();
        for (int i = 0; i < M; i += 4)
            vst1q_f32(y + i, exp_neon(vld1q_f32(x + i)));
        double t1 = now();
        for (int i = 0; i < M; i++)
            y[i] = expf(x[i]);
        double t2 = now();
        if (t1 - t0 < tn) tn = t1 - t0;
        if (t2 - t1 < tl) tl = t2 - t1;
    }
    printf("time per element:  exp_neon %.3f ns   libm expf %.3f ns   (%.1fx)\n", tn / M * 1e9,
           tl / M * 1e9, tl / tn);
    free(x);
    free(y);
    return 0;
}
