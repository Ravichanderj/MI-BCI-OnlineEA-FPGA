#include "hanet_cnn.h"
#include <ap_int.h>
#ifndef HANET_EXPORT_DIR
#include "export_S1/hanet_weights.h"
#include "export_S1/pwl_tables.h"
#include "export_S1/hanet_consts.h"
#endif

typedef ap_int<48> w48;   // all products fit in 48 bits -> 2 DSP48E1 per multiplier

static inline int32_t sat8(w48 v) { return v > 127 ? 127 : (v < -127 ? -127 : (int32_t)v); }

// round-half-up arithmetic shift (sh >= 1)
static inline w48 rshift_round(w48 v, int sh) { return (v + ((w48)1 << (sh - 1))) >> sh; }

static inline w48 mulM(int32_t acc, int32_t M) { return (ap_int<32>)acc * (ap_uint<16>)M; }   // 32x16 multiply

int8_t requant8(int64_t acc, int32_t M, int sh) { return (int8_t)sat8(rshift_round(mulM((int32_t)acc, M), sh)); }

// HW-ELU on a Q.8 value expressed in units of the activation scale s:
//   x>0 : x ;   x<=0 : 2^(x*s*log2 e) - 1   (8-segment PWL of 2^f + barrel shift, as in approx.py)
int32_t elu_q8(int32_t v8, int32_t k16, int32_t inv16) {
#pragma HLS INLINE
    if (v8 > 0) return v8;
    w48 u16 = ((ap_int<24>)v8 * (ap_uint<16>)k16) >> 8;            // 24x16 multiply; exponent Q.16, <= 0
    int32_t i = (int32_t)(u16 >> 16);               // floor
    int32_t f = (int32_t)(u16 & 0xFFFF);            // fraction, Q.16
    int seg = f >> 13, fr = f & 0x1FFF;
    int32_t p14 = EXP_KNOTS_Q14[seg] + (int32_t)(((ap_int<20>)(EXP_KNOTS_Q14[seg + 1] - EXP_KNOTS_Q14[seg]) * (ap_int<20>)fr) >> 13);
    int32_t e14 = (-i >= 16) ? 0 : (p14 >> (-i));  // 2^u in Q.14
    ap_int<18> neg14 = (ap_int<18>)e14 - 16384;     // 2^u - 1 in Q.14 (<= 0)
    return (int32_t)((neg14 * (ap_uint<24>)inv16) >> 22);         // 18x24 multiply; back to Q.8
}

// ---------------------------------------------------------------- v5 parallel layers
// Every layer keeps the integer arithmetic of v4 (integer sums are exact, so reordering the
// additions does not change any result); only the loop structure changes.

// 1. spatial mixing: 22 products per output (x partitioned by channel), II=2 -> 11 multipliers
static void spat_layer(const int8_t x[CNN_C][CNN_T], int8_t spat[CNN_F2][CNN_T]) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=Ws_q complete dim=2
#pragma HLS ALLOCATION instances=mul limit=12 operation
SPAT_T:
    for (int t = 0; t < CNN_T; t++) {
    SPAT_F:
        for (int f = 0; f < CNN_F2; f++) {
#pragma HLS PIPELINE II=2
            int32_t acc = 0;
            for (int c = 0; c < CNN_C; c++) {
#pragma HLS UNROLL
                acc += (int32_t)Ws_q[f][c] * x[c][t];
            }
            spat[f][t] = requant8(acc, Ws_M[f], Ws_sh[f]);
        }
    }
}

// 2. depthwise temporal conv K1=64 ('same', 31 left / 32 right) as a sliding window:
//    at step s the window holds spat[s-63 .. s]; output t = s - 32.  64 taps at II=4 -> 16 mults.
static void temp_layer(const int8_t spat[CNN_F2][CNN_T], int8_t blk1[CNN_F2][CNN_T1]) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=Wt_q complete dim=2
#pragma HLS ALLOCATION instances=mul limit=19 operation
TEMP_F:
    for (int f = 0; f < CNN_F2; f++) {
        int8_t win[CNN_K1];
#pragma HLS ARRAY_PARTITION variable=win complete
        for (int k = 0; k < CNN_K1; k++) {
#pragma HLS UNROLL
            win[k] = 0;
        }
        int32_t psum = 0;
    TEMP_S:
        for (int s = 0; s < CNN_T + CNN_K1 / 2; s++) {
#pragma HLS PIPELINE II=4
            for (int k = 0; k < CNN_K1 - 1; k++) {
#pragma HLS UNROLL
                win[k] = win[k + 1];
            }
            win[CNN_K1 - 1] = (s < CNN_T) ? spat[f][s] : (int8_t)0;
            if (s >= CNN_K1 / 2) {
                const int t = s - CNN_K1 / 2;
                int32_t acc = bt_q[f];
                for (int k = 0; k < CNN_K1; k++) {
#pragma HLS UNROLL
                    acc += (int32_t)Wt_q[f][k] * win[k];
                }
                int32_t v8 = (int32_t)rshift_round(mulM(acc, Wt_M[f]), Wt_sh[f] - 8);
                psum += elu_q8(v8, ELU_K16_BLK1, ELU_INV16_BLK1);
                if ((t & (CNN_P1 - 1)) == CNN_P1 - 1) {
                    blk1[f][t / CNN_P1] = (int8_t)sat8(rshift_round(psum, 10));   // /4 pool, /256 Q.8
                    psum = 0;
                }
            }
        }
    }
}

// 3. depthwise conv K2=16 ('same', 7 left / 8 right) as a sliding window; 16 taps at II=2 -> 8 mults
static void dw_layer(const int8_t blk1[CNN_F2][CNN_T1], int8_t dw[CNN_F2][CNN_T1]) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=Wdw_q complete dim=2
#pragma HLS ALLOCATION instances=mul limit=9 operation
DW_F:
    for (int f = 0; f < CNN_F2; f++) {
        int8_t win[CNN_K2];
#pragma HLS ARRAY_PARTITION variable=win complete
        for (int k = 0; k < CNN_K2; k++) {
#pragma HLS UNROLL
            win[k] = 0;
        }
    DW_S:
        for (int s = 0; s < CNN_T1 + CNN_K2 / 2; s++) {
#pragma HLS PIPELINE II=2
            for (int k = 0; k < CNN_K2 - 1; k++) {
#pragma HLS UNROLL
                win[k] = win[k + 1];
            }
            win[CNN_K2 - 1] = (s < CNN_T1) ? blk1[f][s] : (int8_t)0;
            if (s >= CNN_K2 / 2) {
                int32_t acc = 0;
                for (int k = 0; k < CNN_K2; k++) {
#pragma HLS UNROLL
                    acc += (int32_t)Wdw_q[f][k] * win[k];
                }
                dw[f][s - CNN_K2 / 2] = requant8(acc, Wdw_M[f], Wdw_sh[f]);
            }
        }
    }
}

// 4. pointwise 16x16 + bias -> HW-ELU -> avg-pool 8; 16 products per output at II=2 -> 8 mults
static void pw_layer(const int8_t dw[CNN_F2][CNN_T1], int8_t tok[CNN_F2][CNN_L]) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=Wpw_q complete dim=2
#pragma HLS ALLOCATION instances=mul limit=12 operation
PW_G:
    for (int g = 0; g < CNN_F2; g++) {
        int32_t psum = 0;
    PW_T:
        for (int t = 0; t < CNN_L * CNN_P2; t++) {
#pragma HLS PIPELINE II=2
            int32_t acc = bpw_q[g];
            for (int f = 0; f < CNN_F2; f++) {
#pragma HLS UNROLL
                acc += (int32_t)Wpw_q[g][f] * dw[f][t];
            }
            int32_t v8 = (int32_t)rshift_round(mulM(acc, Wpw_M[g]), Wpw_sh[g] - 8);
            psum += elu_q8(v8, ELU_K16_TOK, ELU_INV16_TOK);
            if ((t & (CNN_P2 - 1)) == CNN_P2 - 1) {
                tok[g][t / CNN_P2] = (int8_t)sat8(rshift_round(psum, 11));     // /8 pool, /256 Q.8
                psum = 0;
            }
        }
    }
}

void hanet_cnn(const int8_t x[CNN_C][CNN_T], int32_t logits[CNN_NCLS]) {
    static int8_t spat[CNN_F2][CNN_T];
    static int8_t blk1[CNN_F2][CNN_T1];
    static int8_t dw[CNN_F2][CNN_T1];
#pragma HLS ARRAY_PARTITION variable=dw complete dim=1
    static int8_t tok[CNN_F2][CNN_L];
    spat_layer(x, spat);
    temp_layer(spat, blk1);
    dw_layer(blk1, dw);
    pw_layer(dw, tok);

// 5. classifier: flatten order (token l, feature f) -> index l*16 + f
HEAD:
    for (int n = 0; n < CNN_NCLS; n++) {
        int32_t acc = bh_q[n];
        for (int l = 0; l < CNN_L; l++)
            for (int f = 0; f < CNN_F2; f++) {
#pragma HLS PIPELINE II=1
                acc += (int32_t)Wh_q[n][l * CNN_F2 + f] * tok[f][l];
            }
        logits[n] = acc;
    }
}
