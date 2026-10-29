/* ARM Cortex-A9 software baseline for the MI-BCI accelerator (same pipeline, same weights).
 *
 *   online EA (double precision, identical algorithm to python/ea.py::OnlineEA and to the
 *   HLS double reference)  ->  int8 input  ->  integer CNN (bit-identical to hls/hanet_cnn.cpp)
 *
 * Build on the PYNQ-Z2 (folder must contain export_S1/ with the 4 headers):
 *   gcc -O3 -mcpu=cortex-a9 -mfpu=neon -mfloat-abi=hard -o cpu_baseline cpu_baseline.c -lm
 * Run (normally through cpu_baseline.py, which writes board_S1.bin from board_S1.npz):
 *   ./cpu_baseline board_S1.bin cpu_pred_S1.txt
 *   ./cpu_baseline --selftest            (checks the integer CNN against golden.h)
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "export_S1/hanet_weights.h"
#include "export_S1/pwl_tables.h"
#include "export_S1/hanet_consts.h"
#include "export_S1/golden.h"

#define C 22
#define T 500
#define F2 16
#define K1 64
#define K2 16
#define P1 4
#define P2 8
#define T1 (T / P1)
#define L (T1 / P2)
#define NCLS 4
#define KEMA 5
#define NS_ITERS 18
#define EPS 0.01

/* ------------------------------------------------------------------ timing */
static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec * 1e-6;
}

/* ------------------------------------------------------------------ online EA (double) */
static double R[C][C];
static int n_seen;

static void ea_reset(void) { memset(R, 0, sizeof R); n_seen = 0; }

static void matmul(double A[C][C], double B[C][C], double O[C][C]) {
    for (int i = 0; i < C; i++)
        for (int j = 0; j < C; j++) {
            double s = 0;
            for (int k = 0; k < C; k++) s += A[i][k] * B[k][j];
            O[i][j] = s;
        }
}

static double scalar_inv_sqrt(double s) {      /* leading-one seed + 6 Newton steps */
    double e = floor(log2(s));
    double r = pow(2.0, -ceil(e / 2));
    for (int k = 0; k < 6; k++) r = r * (1.5 - 0.5 * s * r * r);
    return r;
}

/* x: one trial in uV (C x T) -> W (C x C) */
static void ea_step(const double x[C][T], double W[C][C]) {
    static double P[C][C], S[C][C], Y[C][C], Z[C][C], Tm[C][C], M[C][C], Yn[C][C], Zn[C][C];
    for (int i = 0; i < C; i++)                         /* trial covariance (symmetric) */
        for (int j = i; j < C; j++) {
            double s = 0;
            for (int t = 0; t < T; t++) s += x[i][t] * x[j][t];
            P[i][j] = P[j][i] = s / T;
        }
    n_seen++;
    double a = 1.0 / n_seen, amin = 1.0 / (1 << KEMA);
    if (a < amin) a = amin;
    double tr = 0;
    for (int i = 0; i < C; i++) for (int j = 0; j < C; j++) R[i][j] += a * (P[i][j] - R[i][j]);
    for (int i = 0; i < C; i++) tr += R[i][i];
    double sh = EPS * tr / C, trS = tr + C * sh;
    for (int i = 0; i < C; i++) for (int j = 0; j < C; j++) S[i][j] = R[i][j] + (i == j ? sh : 0);
    double r = scalar_inv_sqrt(trS);
    for (int i = 0; i < C; i++)
        for (int j = 0; j < C; j++) { Y[i][j] = r * r * S[i][j]; Z[i][j] = (i == j); }
    for (int it = 0; it < NS_ITERS; it++) {            /* coupled Newton-Schulz */
        matmul(Z, Y, M);
        for (int i = 0; i < C; i++) for (int j = 0; j < C; j++) Tm[i][j] = 0.5 * ((i == j) * 3.0 - M[i][j]);
        matmul(Y, Tm, Yn);
        matmul(Tm, Z, Zn);
        memcpy(Y, Yn, sizeof Y); memcpy(Z, Zn, sizeof Z);
    }
    for (int i = 0; i < C; i++) for (int j = 0; j < C; j++) W[i][j] = r * 0.5 * (Z[i][j] + Z[j][i]);
}

static void ea_apply_q8(const double W[C][C], const double x[C][T], int8_t q[C][T]) {
    const double g = 1.0 / S_IN;
    for (int i = 0; i < C; i++) {
        double w[C];
        for (int j = 0; j < C; j++) w[j] = W[i][j] * g;
        for (int t = 0; t < T; t++) {
            double s = 0;
            for (int j = 0; j < C; j++) s += w[j] * x[j][t];
            long v = lround(s);
            q[i][t] = (int8_t)(v > 127 ? 127 : (v < -127 ? -127 : v));
        }
    }
}

/* ------------------------------------------------------------------ integer CNN (== hanet_cnn.cpp) */
static inline int32_t sat8(int64_t v) { return v > 127 ? 127 : (v < -127 ? -127 : (int32_t)v); }
static inline int64_t rshift_round(int64_t v, int sh) { return (v + ((int64_t)1 << (sh - 1))) >> sh; }
static inline int64_t mulM(int32_t acc, int32_t M) { return (int64_t)acc * (int64_t)M; }
static inline int8_t requant8(int64_t acc, int32_t M, int sh) { return (int8_t)sat8(rshift_round(mulM((int32_t)acc, M), sh)); }

static int32_t elu_q8(int32_t v8, int32_t k16, int32_t inv16) {
    if (v8 > 0) return v8;
    int64_t u16 = ((int64_t)v8 * k16) >> 8;
    int32_t i = (int32_t)(u16 >> 16);
    int32_t f = (int32_t)(u16 & 0xFFFF);
    int seg = f >> 13, fr = f & 0x1FFF;
    int32_t p14 = EXP_KNOTS_Q14[seg] + (int32_t)(((int64_t)(EXP_KNOTS_Q14[seg + 1] - EXP_KNOTS_Q14[seg]) * fr) >> 13);
    int32_t e14 = (-i >= 16) ? 0 : (p14 >> (-i));
    int32_t neg14 = e14 - 16384;
    return (int32_t)(((int64_t)neg14 * inv16) >> 22);
}

static void cnn(const int8_t x[C][T], int32_t logits[NCLS]) {
    static int8_t spat[F2][T], blk1[F2][T1], dw[F2][T1], tok[F2][L];
    for (int t = 0; t < T; t++)
        for (int f = 0; f < F2; f++) {
            int32_t acc = 0;
            for (int c = 0; c < C; c++) acc += (int32_t)Ws_q[f][c] * x[c][t];
            spat[f][t] = requant8(acc, Ws_M[f], Ws_sh[f]);
        }
    for (int f = 0; f < F2; f++)
        for (int t1 = 0; t1 < T1; t1++) {
            int32_t psum = 0;
            for (int p = 0; p < P1; p++) {
                int t = t1 * P1 + p;
                int32_t acc = bt_q[f];
                for (int k = 0; k < K1; k++) {
                    int ti = t + k - (K1 - 1) / 2;
                    if (ti >= 0 && ti < T) acc += (int32_t)Wt_q[f][k] * spat[f][ti];
                }
                int32_t v8 = (int32_t)rshift_round(mulM(acc, Wt_M[f]), Wt_sh[f] - 8);
                psum += elu_q8(v8, ELU_K16_BLK1, ELU_INV16_BLK1);
            }
            blk1[f][t1] = (int8_t)sat8(rshift_round(psum, 10));
        }
    for (int f = 0; f < F2; f++)
        for (int t = 0; t < T1; t++) {
            int32_t acc = 0;
            for (int k = 0; k < K2; k++) {
                int ti = t + k - (K2 - 1) / 2;
                if (ti >= 0 && ti < T1) acc += (int32_t)Wdw_q[f][k] * blk1[f][ti];
            }
            dw[f][t] = requant8(acc, Wdw_M[f], Wdw_sh[f]);
        }
    for (int g = 0; g < F2; g++)
        for (int l = 0; l < L; l++) {
            int32_t psum = 0;
            for (int p = 0; p < P2; p++) {
                int t = l * P2 + p;
                int32_t acc = bpw_q[g];
                for (int f = 0; f < F2; f++) acc += (int32_t)Wpw_q[g][f] * dw[f][t];
                int32_t v8 = (int32_t)rshift_round(mulM(acc, Wpw_M[g]), Wpw_sh[g] - 8);
                psum += elu_q8(v8, ELU_K16_TOK, ELU_INV16_TOK);
            }
            tok[g][l] = (int8_t)sat8(rshift_round(psum, 11));
        }
    for (int n = 0; n < NCLS; n++) {
        int32_t acc = bh_q[n];
        for (int l = 0; l < L; l++)
            for (int f = 0; f < F2; f++) acc += (int32_t)Wh_q[n][l * F2 + f] * tok[f][l];
        logits[n] = acc;
    }
}

static int argmax_w(const int32_t a[NCLS]) {
    int b = 0;
    int64_t bv = (int64_t)a[0] * LOGIT_W[0];
    for (int n = 1; n < NCLS; n++) {
        int64_t v = (int64_t)a[n] * LOGIT_W[n];
        if (v > bv) { bv = v; b = n; }
    }
    return b;
}

/* ------------------------------------------------------------------ main */
static int selftest(void) {
    int32_t lg[NCLS];
    cnn(GOLDEN_IN, lg);
    double err = 0, mag = 0;
    for (int n = 0; n < NCLS; n++) {
        double v = lg[n] * (double)LOGIT_SCALE[n];
        printf("logit %d: C %.6f  golden %.6f\n", n, v, GOLDEN_LOGITS[n]);
        err += fabs(v - GOLDEN_LOGITS[n]); mag += fabs(GOLDEN_LOGITS[n]);
    }
    int ok = err / mag < 0.05;
    printf("selftest %s (rel. diff %.3f%%)\n", ok ? "PASS" : "FAIL", 100 * err / mag);
    return ok ? 0 : 1;
}

int main(int argc, char **argv) {
    if (argc > 1 && strcmp(argv[1], "--selftest") == 0) return selftest();
    if (argc < 3) { fprintf(stderr, "usage: %s data.bin pred.txt | --selftest\n", argv[0]); return 2; }
    FILE *fi = fopen(argv[1], "rb");
    if (!fi) { perror(argv[1]); return 2; }
    int32_t n;                                       /* format: n, then per trial: session, label, C*T int32 Q.10 */
    if (fread(&n, 4, 1, fi) != 1) return 2;
    FILE *fo = fopen(argv[2], "w");
    static int32_t raw[C * T];
    static double x[C][T], W[C][C];
    static int8_t q[C][T];
    int prev = -1, correct = 0;
    double t_ea = 0, t_cnn = 0, t_max = 0, t_min = 1e9;
    for (int k = 0; k < n; k++) {
        int32_t hdr[2];
        if (fread(hdr, 4, 2, fi) != 2 || fread(raw, 4, C * T, fi) != C * T) { fprintf(stderr, "short read\n"); return 2; }
        for (int i = 0; i < C; i++) for (int t = 0; t < T; t++) x[i][t] = raw[i * T + t] / 1024.0;
        double t0 = now_ms();                         /* timed: everything the FPGA does per trial */
        if (hdr[0] != prev) { ea_reset(); prev = hdr[0]; }
        ea_step(x, W);
        ea_apply_q8(W, x, q);
        double t1 = now_ms();
        int32_t lg[NCLS];
        cnn(q, lg);
        int p = argmax_w(lg);
        double t2 = now_ms();
        t_ea += t1 - t0; t_cnn += t2 - t1;
        double tt = t2 - t0; if (tt > t_max) t_max = tt; if (tt < t_min) t_min = tt;
        correct += (p == hdr[1]);
        fprintf(fo, "%d %.4f %.4f\n", p, t1 - t0, t2 - t1);
        if (k % 100 == 0) { printf("trial %d: pred %d true %d  %.2f ms\n", k, p, hdr[1], tt); fflush(stdout); }
    }
    fclose(fi); fclose(fo);
    printf("\nARM Cortex-A9 (1 core) | %d trials | accuracy %.2f%% | latency/trial mean %.2f ms "
           "(EA %.2f + CNN %.2f) [min %.2f, max %.2f]\n",
           n, 100.0 * correct / n, (t_ea + t_cnn) / n, t_ea / n, t_cnn / n, t_min, t_max);
    return 0;
}
