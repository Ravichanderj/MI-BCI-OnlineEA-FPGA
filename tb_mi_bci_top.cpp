// End-to-end C-simulation of the accelerator top on a 60-trial stream (ea_in.bin, from
// make_ea_testdata.py).  Reference chain = double-precision online EA -> same integer CNN.
// Checks: (1) int8 decoder inputs agree, (2) predicted classes agree, (3) golden CNN vector.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include "online_ea.h"
#include "hanet_cnn.h"
#include "export_S1/hanet_consts.h"
#include "export_S1/golden.h"

void mi_bci_top(const int32_t xin[EA_C * EA_T], int32_t *l0, int32_t *l1, int32_t *l2, int32_t *l3,
                int32_t *cls, int32_t reset);

static int argmax(const int32_t a[CNN_NCLS]) {
    int b = 0;
    for (int n = 1; n < CNN_NCLS; n++) if (a[n] * LOGIT_SCALE[n] > a[b] * LOGIT_SCALE[b]) b = n;
    return b;
}

int main() {
    // (3) decoder alone on the deploy.py golden vector
    int32_t g[CNN_NCLS];
    hanet_cnn(GOLDEN_IN, g);
    int gref = 0;
    for (int n = 1; n < CNN_NCLS; n++) if (GOLDEN_LOGITS[n] > GOLDEN_LOGITS[gref]) gref = n;
    bool golden_ok = argmax(g) == gref;
    printf("golden vector: class HLS %d / python %d\n", argmax(g), gref);

    FILE *f = fopen("ea_in.bin", "rb");
    if (!f) { printf("run: python3 make_ea_testdata.py first\n"); return 1; }
    fseek(f, 0, SEEK_END); long bytes = ftell(f); fseek(f, 0, SEEK_SET);
    int n = bytes / (sizeof(float) * EA_C * EA_T);
    std::vector<float> buf(bytes / sizeof(float));
    if (fread(buf.data(), sizeof(float), buf.size(), f) != buf.size()) return 1;
    fclose(f);

    static int32_t xin[EA_C * EA_T];
    static double xd[EA_C][EA_T];
    static int8_t qref[EA_C][EA_T];
    int agree_cls = 0; long diff_q = 0, big_q = 0;
    for (int k = 0; k < n; k++) {
        for (int i = 0; i < EA_C; i++)
            for (int t = 0; t < EA_T; t++) {
                float v = buf[(size_t)k * EA_C * EA_T + i * EA_T + t];
                xin[i * EA_T + t] = (int32_t)lround(v * 1024.0);   // Q.10, as the ARM driver sends
                xd[i][t] = xin[i * EA_T + t] / 1024.0;
            }
        int32_t lg[CNN_NCLS], cls;
        mi_bci_top(xin, &lg[0], &lg[1], &lg[2], &lg[3], &cls, k == 0);
        online_ea_q8_ref(xd, qref, 1.0 / S_IN, k == 0);
        int32_t lr[CNN_NCLS];
        hanet_cnn(qref, lr);
        agree_cls += (cls == argmax(lr));
        for (int c = 0; c < CNN_NCLS; c++) { diff_q += labs((long)lg[c] - lr[c]); big_q += labs((long)lr[c]); }
    }
    double rel = (double)diff_q / (double)big_q;
    printf("stream of %d trials: class agreement fixed-point vs double EA = %d/%d, rel. logit diff %.3f%%\n",
           n, agree_cls, n, 100 * rel);
    bool ok = golden_ok && agree_cls >= n - 1 && rel < 0.05;
    printf(ok ? "PASS\n" : "FAIL\n");
    return ok ? 0 : 1;
}
