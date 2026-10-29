// Online Euclidean Alignment — HLS kernel, v5 (parallel datapaths). See python/ea.py for the math.
//
//   P_n = X X^T / T                 (upper triangle only)
//   R_n = R + a_n (P_n - R),        a_n = max(1/n, 2^-k)  (33-entry ROM, then constant 2^-k)
//   S_n = R_n + eps tr(R_n)/C I
//   r_n = tr(S_n)^{-1/2}            leading-one detector -> tr = m 2^e, m in [1,2);
//                                   r = m^{-1/2} (fixed-point Newton) * 2^{-e/2}  (shift)
//   A_n = r^2 S_n                   eigenvalues in (0,1]
//   Z_n = A^{-1/2}                  coupled Newton-Schulz, EA_NS_ITERS iterations
//   Y   = r Z X
// No floating point, no divider, no square-root unit.  The core is templated so the SAME
// code runs in double (reference, test bench only) and in ap_fixed (hardware).
#include "online_ea.h"
#include <cmath>

// narrow constant type for multiplications by constants <= 1 (double in the reference)
template <typename X> struct Narrow { typedef X T; typedef X P; };
template <> struct Narrow<cov_t> { typedef ap_ufixed<26, 1> T; typedef ap_fixed<80, 30> P; };

// trial covariance normalisation.  Reference: s / T.  Hardware: s / 512 (a shift, no
// multiplier); R is then 500/512 of the true mean, which scales r by sqrt(512/500) and is
// compensated exactly by EA_COMP in the output gain (shrinkage and A are scale-invariant).
template <typename COV, typename CACC> static inline COV trial_mean(CACC s) { return (COV)(s / EA_T); }
template <> inline cov_t trial_mean<cov_t, cacc_t>(cacc_t s) { return (cov_t)(s >> 9); }

// ------------------------------------------------------------------ 1/n ROM
template <typename COV>
static COV alpha_of(int n) {
    static const COV tab[(1 << EA_KEMA) + 1] = {0.0, 1.0, 0.5, 0.3333333333333333, 0.25, 0.2, 0.16666666666666666, 0.14285714285714285, 0.125, 0.1111111111111111, 0.1, 0.09090909090909091, 0.08333333333333333, 0.07692307692307693, 0.07142857142857142, 0.06666666666666667, 0.0625, 0.058823529411764705, 0.05555555555555555, 0.05263157894736842, 0.05, 0.047619047619047616, 0.045454545454545456, 0.043478260869565216, 0.041666666666666664, 0.04, 0.038461538461538464, 0.037037037037037035, 0.03571428571428571, 0.034482758620689655, 0.03333333333333333, 0.03225806451612903, 0.03125};
    return tab[n];
}

// x/2: a wire shift in hardware, plain multiply in the double reference
// explicit type argument (no overload resolution: Xilinx ap_fixed converts implicitly to double)
template <typename T> static inline T half_t(T v) { return v * (T)0.5; }
template <> inline mat_t half_t<mat_t>(mat_t v) { return v >> 1; }
template <> inline acc_t half_t<acc_t>(acc_t v) { return v >> 1; }

template <typename T> static inline T mul22(T v) { return v * (T)22; }
template <> inline cov_t mul22<cov_t>(cov_t v) { return (cov_t)((v << 4) + (v << 2) + (v << 1)); }

// ------------------------------------------------------------------ Newton-Schulz
template <typename MAT, typename ACC>
static void matmul(const MAT A[EA_C][EA_C], const MAT B[EA_C][EA_C], MAT Cm[EA_C][EA_C]) {
#pragma HLS INLINE off
#pragma HLS ALLOCATION instances=mul limit=11 operation
    // v5: row-accumulate form  C[i][:] += A[i][k] * B[k][:]  -> reads one row of B per step
    // (all matrices are partitioned by column).  The per-element summation order over k is the
    // same as in v4, so the result is bit-identical.  22 products per step at II=2 -> 11 multipliers.
MM_I:
    for (int i = 0; i < EA_C; i++) {
        ACC s[EA_C];
#pragma HLS ARRAY_PARTITION variable=s complete
        for (int j = 0; j < EA_C; j++) {
#pragma HLS UNROLL
            s[j] = 0;
        }
    MM_K:
        for (int k = 0; k < EA_C; k++) {
#pragma HLS PIPELINE II=2
            const MAT a = A[i][k];
            for (int j = 0; j < EA_C; j++) {
#pragma HLS UNROLL
                s[j] += a * B[k][j];
            }
        }
        for (int j = 0; j < EA_C; j++) {
#pragma HLS UNROLL
            Cm[i][j] = (MAT)s[j];
        }
    }
}

template <typename MAT, typename ACC>
static void coupled_ns(const MAT A[EA_C][EA_C], MAT Z[EA_C][EA_C]) {
    MAT Y[EA_C][EA_C], T[EA_C][EA_C], M[EA_C][EA_C], tmp[EA_C][EA_C], Zi[EA_C][EA_C];
#pragma HLS ARRAY_PARTITION variable=Y complete dim=2
#pragma HLS ARRAY_PARTITION variable=T complete dim=2
#pragma HLS ARRAY_PARTITION variable=M complete dim=2
#pragma HLS ARRAY_PARTITION variable=tmp complete dim=2
#pragma HLS ARRAY_PARTITION variable=Zi complete dim=2
INIT:
    for (int i = 0; i < EA_C; i++) {
#pragma HLS PIPELINE II=1
        for (int j = 0; j < EA_C; j++) {
            Y[i][j] = A[i][j]; Zi[i][j] = (i == j) ? (MAT)1 : (MAT)0;
        }
    }
NS_ITER:
    for (int it = 0; it < EA_NS_ITERS; it++) {
        matmul<MAT, ACC>(Zi, Y, M);                                   // M = Z Y
    MK_T:
        for (int i = 0; i < EA_C; i++) {
#pragma HLS PIPELINE II=1
            for (int j = 0; j < EA_C; j++)                            // T = 1.5 I - 0.5 M
                T[i][j] = (MAT)((i == j ? (MAT)1.5 : (MAT)0) - half_t<MAT>(M[i][j]));   // shift, no multiply
        }
        matmul<MAT, ACC>(Y, T, tmp);                                  // Y = Y T
    CP_Y:
        for (int i = 0; i < EA_C; i++) {
#pragma HLS PIPELINE II=1
            for (int j = 0; j < EA_C; j++) Y[i][j] = tmp[i][j];
        }
        matmul<MAT, ACC>(T, Zi, tmp);                                 // Z = T Z
    CP_Z:
        for (int i = 0; i < EA_C; i++) {
#pragma HLS PIPELINE II=1
            for (int j = 0; j < EA_C; j++) Zi[i][j] = tmp[i][j];
        }
    }
SYM:   // Z = (Z + Z^T) / 2  (same values as the v4 in-place version)
    for (int i = 0; i < EA_C; i++)
        for (int j = 0; j < EA_C; j++) {
#pragma HLS PIPELINE II=1
            Z[i][j] = (i == j) ? Zi[i][i] : (MAT)half_t<ACC>((ACC)((ACC)Zi[i][j] + (ACC)Zi[j][i]));
        }
}

// ------------------------------------------------------------------ scalar normalisation
// double reference
static void normalise(const double R[EA_C][EA_C], double sh, double trS, double A[EA_C][EA_C],
                      RScale<double> &rs) {
    double r = 1.0 / std::sqrt(trS);
    for (int i = 0; i < EA_C; i++)
        for (int j = 0; j < EA_C; j++) A[i][j] = (R[i][j] + (i == j ? sh : 0.0)) * r * r;
    rs.rm = r; rs.h = 0;
}

// hardware: leading-one detector + 6 fixed-point Newton steps on the mantissa + shifts
static void normalise(const cov_t R[EA_C][EA_C], cov_t sh, cov_t trS, mat_t A[EA_C][EA_C],
                      RScale<nr_t> &rs) {
    int msb = 0;
LOD:   // sequential (48 cycles once per trial) -> no long combinational priority chain
    for (int b = 0; b < 48; b++) {
        if (trS[b]) msb = b;
    }
    int e = msb - 24;                                   // trS in [2^e, 2^(e+1))
    ap_ufixed<26, 2> m = (e >= 0) ? (ap_ufixed<26, 2>)(trS >> e) : (ap_ufixed<26, 2>)(trS << (-e));
    ap_ufixed<26, 1> r = 0.85;                          // m^{-1/2} in (0.707, 1]
NEWTON:   // recurrence: NOT pipelined, so each 26x26 multiply gets its own clock cycle
    for (int k = 0; k < 6; k++) {
        ap_ufixed<26, 2> r2 = r * r;
        ap_ufixed<26, 2> mr2 = m * r2;
        ap_ufixed<26, 1> c = (ap_ufixed<26, 2>)1.5 - (ap_ufixed<26, 2>)(mr2 >> 1);
        r = r * c;
    }
    int h = e >> 1;                                     // floor(e/2)
    if (e & 1) r = r * (ap_ufixed<26, 1>)0.7071067811865476;
    ap_ufixed<25, 1> rr2 = r * r;                       // r_m^2
    const int s2 = 2 * h;                               // A = S * r_m^2 * 2^-2h
NORM:
    for (int i = 0; i < EA_C; i++)
        for (int j = 0; j < EA_C; j++) {
#pragma HLS PIPELINE II=1
            cov_t s = R[i][j] + (i == j ? sh : (cov_t)0);
            ap_fixed<32, 4> ss = (s2 >= 0) ? (ap_fixed<32, 4>)(s >> s2) : (ap_fixed<32, 4>)(s << (-s2));
            A[i][j] = (mat_t)(ss * rr2);                // 32x25 multiply
        }
    rs.rm = (nr_t)r; rs.h = h;
}

// ------------------------------------------------------------------ trial covariance
// v5: for each row i, 22 accumulators over t (x is partitioned by channel), II=2 -> 11 multipliers.
// Fixed-point sums are exact, so the result equals the v4 pairwise loop bit for bit.
template <typename DIN, typename COV, typename CACC>
static void cov_rows(const DIN x[EA_C][EA_T], COV P[EA_C][EA_C]) {
#pragma HLS INLINE off
#pragma HLS ALLOCATION instances=mul limit=11 operation
COV_I:
    for (int i = 0; i < EA_C; i++) {
        CACC s[EA_C];
#pragma HLS ARRAY_PARTITION variable=s complete
        for (int j = 0; j < EA_C; j++) {
#pragma HLS UNROLL
            s[j] = 0;
        }
    COV_T:
        for (int t = 0; t < EA_T; t++) {
#pragma HLS PIPELINE II=2
            const DIN xi = x[i][t];
            for (int j = 0; j < EA_C; j++) {
#pragma HLS UNROLL
                s[j] += xi * x[j][t];          // native 24x24 product
            }
        }
        for (int j = 0; j < EA_C; j++) {
#pragma HLS UNROLL
            P[i][j] = trial_mean<COV, CACC>(s[j]);    // HW: s / 512 (shift); scale fixed in gain()
        }
    }
}

// ------------------------------------------------------------------ streaming core
template <typename DIN, typename COV, typename MAT, typename ACC, typename CACC, typename RT>
struct OnlineEACore {
    COV R[EA_C][EA_C];
    int n;
    void reset() {
        n = 0;
        for (int i = 0; i < EA_C; i++)
            for (int j = 0; j < EA_C; j++) R[i][j] = 0;
    }

    void update(const DIN x[EA_C][EA_T], MAT Z[EA_C][EA_C], RScale<RT> &rs) {
        // 1) trial covariance (upper triangle) and recursive reference update
        n = (n < (1 << EA_KEMA)) ? n + 1 : n;
        const typename Narrow<COV>::T a = alpha_of<typename Narrow<COV>::T>(n);   // exact mean, then 2^-k
        COV P[EA_C][EA_C];
        cov_rows<DIN, COV, CACC>(x, P);
    UPD:   // upper triangle only; each element is read and written once -> no loop-carried path
        for (int i = 0; i < EA_C; i++)
            for (int j = 0; j < EA_C; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS DEPENDENCE variable=R inter false
                if (j >= i) R[i][j] = R[i][j] + (COV)((P[i][j] - R[i][j]) * a);
            }
    MIRROR:
        for (int i = 1; i < EA_C; i++)
            for (int j = 0; j < i; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS DEPENDENCE variable=R inter false
                R[i][j] = R[j][i];
            }
        // 2) shrinkage + trace
        COV tr = 0;
    TRACE:
        for (int i = 0; i < EA_C; i++) {
#pragma HLS PIPELINE II=1
            tr += R[i][i];
        }
        typename Narrow<COV>::P sh_p = tr * (typename Narrow<COV>::T)(EA_EPS / EA_C);
#pragma HLS RESOURCE variable=sh_p core=MulnS latency=4
        COV sh = (COV)sh_p;
        COV trS = tr + mul22<COV>(sh);                 // sh*22 with shifts
        // 3)+4) scalar inverse sqrt and A = r^2 S
        MAT A[EA_C][EA_C];
#pragma HLS ARRAY_PARTITION variable=A complete dim=2
        normalise(R, sh, trS, A, rs);
        // 5) Newton-Schulz
        coupled_ns<MAT, ACC>(A, Z);
    }
};

// ------------------------------------------------------------------ hardware entry points
static OnlineEACore<din_t, cov_t, mat_t, acc_t, cacc_t, nr_t> g_ea;

static acc_t gain(const RScale<nr_t> &rs, acc_t c) {      // r * c = r_m * c * 2^-h
    ap_ufixed<54, 9> gp = (ap_ufixed<26, 1>)rs.rm * (ap_ufixed<28, 8>)c;   // 26x28 multiply (c includes EA_COMP)
#pragma HLS RESOURCE variable=gp core=MulnS latency=4
    acc_t g0 = (acc_t)gp;
    return (rs.h >= 0) ? (acc_t)(g0 >> rs.h) : (acc_t)(g0 << (-rs.h));
}

// v5 whitening fused with int8 quantisation: one output sample per 2 cycles, 22 products of
// Z[i][:] x[:][t] (Z partitioned by column, x by channel).  The sum is exact in app_t, so the
// output equals v4 bit for bit.
static void whiten_q8(const din_t x[EA_C][EA_T], const mat_t Z[EA_C][EA_C], acc_t g, int8_t q[EA_C][EA_T]) {
#pragma HLS INLINE off
#pragma HLS ALLOCATION instances=mul limit=12 operation
Q_T:
    for (int t = 0; t < EA_T; t++) {
    Q_I:
        for (int i = 0; i < EA_C; i++) {
#pragma HLS PIPELINE II=2
            app_t s = 0;
            for (int j = 0; j < EA_C; j++) {
#pragma HLS UNROLL
                s += Z[i][j] * x[j][t];          // native 32x24 product
            }
            ap_fixed<40, 20> v = (ap_fixed<36, 24>)s * (ap_ufixed<25, 0>)g;
            ap_fixed<42, 22> vr = (ap_fixed<42, 22>)v + (ap_fixed<42, 22>)0.5;   // round half up
            ap_fixed<22, 22> fi = vr;                                             // AP_TRN = floor
            int iv = fi.to_int();
            q[i][t] = (int8_t)(iv > 127 ? 127 : (iv < -127 ? -127 : (int)iv));
        }
    }
}

// Whitening fused with int8 quantisation of the decoder input:  q = clip(round(r Z x / s_in))
void online_ea_q8(const din_t x[EA_C][EA_T], int8_t q[EA_C][EA_T], acc_t inv_s_in, bool reset) {
#pragma HLS INLINE off
    if (reset) g_ea.reset();
    mat_t Z[EA_C][EA_C];
#pragma HLS ARRAY_PARTITION variable=Z complete dim=2
    RScale<nr_t> rs;
    g_ea.update(x, Z, rs);
    const acc_t g = gain(rs, inv_s_in);     // caller passes EA_COMP / s_in
    whiten_q8(x, Z, g, q);
}

// aligned output (unit variance) — standalone EA IP / test bench
void online_ea(const din_t x[EA_C][EA_T], dout_t y[EA_C][EA_T], bool reset) {
#pragma HLS INTERFACE m_axi port=x offset=slave bundle=gmem0 depth=11000
#pragma HLS INTERFACE m_axi port=y offset=slave bundle=gmem1 depth=11000
#pragma HLS INTERFACE s_axilite port=reset
#pragma HLS INTERFACE s_axilite port=return
    if (reset) g_ea.reset();
    mat_t Z[EA_C][EA_C];
    RScale<nr_t> rs;
    g_ea.update(x, Z, rs);
    const acc_t g = gain(rs, (acc_t)EA_COMP);
APPLY_T:
    for (int t = 0; t < EA_T; t++) {
    APPLY_I:
        for (int i = 0; i < EA_C; i++) {
            app_t s = 0;
            for (int j = 0; j < EA_C; j++) {
#pragma HLS PIPELINE II=1
                s += Z[i][j] * x[j][t];
            }
            y[i][t] = (dout_t)((ap_fixed<36, 24>)s * (ap_ufixed<25, 2>)g);
        }
    }
}

#ifndef __SYNTHESIS__
// ------------------------------------------------------------------ references (test bench only)
void online_ea_q8_ref(const double x[EA_C][EA_T], int8_t q[EA_C][EA_T], double inv_s_in, bool reset) {
    static OnlineEACore<double, double, double, double, double, double> ref;
    if (reset) ref.reset();
    double Z[EA_C][EA_C]; RScale<double> rs;
    ref.update(x, Z, rs);
    for (int t = 0; t < EA_T; t++)
        for (int i = 0; i < EA_C; i++) {
            double s = 0;
            for (int j = 0; j < EA_C; j++) s += Z[i][j] * x[j][t];
            long iv = lround(s * rs.rm * inv_s_in);
            q[i][t] = (int8_t)(iv > 127 ? 127 : (iv < -127 ? -127 : iv));
        }
}

void online_ea_ref(const double x[EA_C][EA_T], double W[EA_C][EA_C], bool reset) {
    static OnlineEACore<double, double, double, double, double, double> ref;
    if (reset) ref.reset();
    double Z[EA_C][EA_C]; RScale<double> rs;
    ref.update(x, Z, rs);
    for (int i = 0; i < EA_C; i++) for (int j = 0; j < EA_C; j++) W[i][j] = Z[i][j] * rs.rm;
}

void online_ea_fixed_W(const din_t x[EA_C][EA_T], double W[EA_C][EA_C], bool reset) {
    static OnlineEACore<din_t, cov_t, mat_t, acc_t, cacc_t, nr_t> fx;
    if (reset) fx.reset();
    mat_t Z[EA_C][EA_C]; RScale<nr_t> rs;
    fx.update(x, Z, rs);
    double r = rs.rm.to_double() * std::pow(2.0, -rs.h) * EA_COMP;
    for (int i = 0; i < EA_C; i++) for (int j = 0; j < EA_C; j++) W[i][j] = Z[i][j].to_double() * r;
}
#endif  // __SYNTHESIS__
