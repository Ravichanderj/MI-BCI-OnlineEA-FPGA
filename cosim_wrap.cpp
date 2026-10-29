// Wrappers for C/RTL co-simulation of the two halves of mi_bci_top, with the SAME internal
// array partitioning as mi_bci_top.cpp (so the same hardware is generated).
#include "online_ea.h"
#include "hanet_cnn.h"
#include "export_S1/hanet_consts.h"

// half 1: online EA + int8 quantisation.  xin: 22x500 Q.10 words, qout: 22x500 int8
void ea_wrap(const int32_t xin[EA_C * EA_T], int8_t qout[EA_C * EA_T], int32_t reset) {
    static din_t x[EA_C][EA_T];
#pragma HLS ARRAY_PARTITION variable=x complete dim=1
    static int8_t q[EA_C][EA_T];
#pragma HLS ARRAY_PARTITION variable=q complete dim=1
LD: for (int i = 0; i < EA_C; i++)
        for (int t = 0; t < EA_T; t++) {
#pragma HLS PIPELINE II=1
            din_t v; v.range() = xin[i * EA_T + t]; x[i][t] = v;
        }
    online_ea_q8(x, q, (acc_t)(EA_COMP / S_IN), reset != 0);
ST: for (int i = 0; i < EA_C; i++)
        for (int t = 0; t < EA_T; t++) {
#pragma HLS PIPELINE II=1
            qout[i * EA_T + t] = q[i][t];
        }
}

// half 2: int8 decoder.  qin: 22x500 int8, logits: 4 x int32
void cnn_wrap(const int8_t qin[CNN_C * CNN_T], int32_t logits[CNN_NCLS]) {
    static int8_t q[CNN_C][CNN_T];
#pragma HLS ARRAY_PARTITION variable=q complete dim=1
LD: for (int i = 0; i < CNN_C; i++)
        for (int t = 0; t < CNN_T; t++) {
#pragma HLS PIPELINE II=1
            q[i][t] = qin[i * CNN_T + t];
        }
    int32_t acc[CNN_NCLS];
    hanet_cnn(q, acc);
    for (int n = 0; n < CNN_NCLS; n++) logits[n] = acc[n];
}
