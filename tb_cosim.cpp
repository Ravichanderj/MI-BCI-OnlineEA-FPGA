// Exact (bit-level) C/RTL co-simulation test bench on the first 2 trials of subject S1.
// Build with -DMODE_TOP, -DMODE_EA or -DMODE_CNN (run_cosim.tcl does this).
// Data: cosim_in.bin   = 2 trials x 11000 int32 (Q.10 input, as streamed by the board)
//       cosim_q.bin    = 2 trials x 11000 int8  (expected EA output = decoder input, C model)
//       cosim_log.bin  = 2 trials x 4 int32     (expected logits, C model)
#include <cstdio>
#include <cstdint>
#include "online_ea.h"
#include "hanet_cnn.h"
void mi_bci_top(const int32_t xin[22 * 500], int32_t *l0, int32_t *l1, int32_t *l2, int32_t *l3, int32_t *cls, int32_t reset);
void ea_wrap(const int32_t xin[22 * 500], int8_t qout[22 * 500], int32_t reset);
void cnn_wrap(const int8_t qin[22 * 500], int32_t logits[4]);

static int32_t X[2][11000]; static int8_t Q[2][11000]; static int32_t L[2][4];
static bool load(const char *n, void *p, size_t sz) {
    FILE *f = fopen(n, "rb"); if (!f) { printf("missing %s\n", n); return false; }
    size_t r = fread(p, 1, sz, f); fclose(f); return r == sz;
}
int main() {
    if (!load("cosim_in.bin", X, sizeof X) || !load("cosim_q.bin", Q, sizeof Q) || !load("cosim_log.bin", L, sizeof L)) return 1;
    int bad = 0;
    for (int k = 0; k < 2; k++) {
#if defined(MODE_TOP)
        int32_t l[4], c;
        mi_bci_top(X[k], &l[0], &l[1], &l[2], &l[3], &c, k == 0);
        printf("TOP trial %d: RTL [%d %d %d %d]  expected [%d %d %d %d]\n", k, l[0], l[1], l[2], l[3], L[k][0], L[k][1], L[k][2], L[k][3]);
        for (int n = 0; n < 4; n++) bad += (l[n] != L[k][n]);
#elif defined(MODE_EA)
        static int8_t q[11000];
        ea_wrap(X[k], q, k == 0);
        int nb = 0, first = -1;
        for (int i = 0; i < 11000; i++) if (q[i] != Q[k][i]) { if (first < 0) first = i; nb++; }
        printf("EA  trial %d: %d of 11000 outputs differ", k, nb);
        if (first >= 0) printf(" (first: ch %d, t %d: RTL %d expected %d)", first / 500, first % 500, q[first], Q[k][first]);
        printf("\n"); bad += nb;
#elif defined(MODE_CNN)
        int32_t l[4];
        cnn_wrap(Q[k], l);
        printf("CNN trial %d: RTL [%d %d %d %d]  expected [%d %d %d %d]\n", k, l[0], l[1], l[2], l[3], L[k][0], L[k][1], L[k][2], L[k][3]);
        for (int n = 0; n < 4; n++) bad += (l[n] != L[k][n]);
#endif
    }
    printf(bad ? "RESULT: MISMATCH\n" : "RESULT: EXACT MATCH\n");
    return bad ? 1 : 0;
}
