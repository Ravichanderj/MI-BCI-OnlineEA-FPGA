// Full accelerator top (Vivado HLS 2019.2): one 4-s trial per call.
//   DDR (int32, filtered EEG in uV, Q.10)  -> online Euclidean alignment (label-free, streaming)
//   -> int8 decoder input -> integer CNN -> 4 logits (l0..l3, int32; real = l*LOGIT_SCALE) + predicted class
// Control/status over AXI-Lite (PYNQ: overlay.mi_bci_top_0.register_map / .write/.read).
#include "online_ea.h"
#include "hanet_cnn.h"
#include "export_S1/hanet_consts.h"

void mi_bci_top(const int32_t xin[EA_C * EA_T], int32_t *l0, int32_t *l1, int32_t *l2, int32_t *l3,
                int32_t *cls, int32_t reset) {
#pragma HLS INTERFACE m_axi port=xin offset=slave bundle=gmem depth=11000
#pragma HLS INTERFACE s_axilite port=xin
#pragma HLS INTERFACE s_axilite port=l0
#pragma HLS INTERFACE s_axilite port=l1
#pragma HLS INTERFACE s_axilite port=l2
#pragma HLS INTERFACE s_axilite port=l3
#pragma HLS INTERFACE s_axilite port=cls
#pragma HLS INTERFACE s_axilite port=reset
#pragma HLS INTERFACE s_axilite port=return
    static din_t x[EA_C][EA_T];
#pragma HLS ARRAY_PARTITION variable=x complete dim=1
    static int8_t q[EA_C][EA_T];
#pragma HLS ARRAY_PARTITION variable=q complete dim=1
LOAD:
    for (int i = 0; i < EA_C; i++)
        for (int t = 0; t < EA_T; t++) {
#pragma HLS PIPELINE II=1
            din_t v;
            v.range() = xin[i * EA_T + t];          // Q.10 fixed point, same bits as din_t
            x[i][t] = v;
        }
    online_ea_q8(x, q, (acc_t)(EA_COMP / S_IN), reset != 0);
    int32_t acc[CNN_NCLS];
    hanet_cnn(q, acc);
    // argmax of real logits = acc * scale_n  ->  integer compare with relative Q15 scales
    int best = 0;
    ap_int<48> bv = (ap_int<32>)acc[0] * (ap_uint<16>)LOGIT_W[0];
OUT:
    for (int n = 1; n < CNN_NCLS; n++) {
        ap_int<48> v = (ap_int<32>)acc[n] * (ap_uint<16>)LOGIT_W[n];
        if (v > bv) { bv = v; best = n; }
    }
    *l0 = acc[0]; *l1 = acc[1]; *l2 = acc[2]; *l3 = acc[3];
    *cls = best;
}
