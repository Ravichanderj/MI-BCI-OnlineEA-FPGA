#include <cstdio>
#include <cstdint>
void mi_bci_top(const int32_t xin[22 * 500], int32_t *l0, int32_t *l1, int32_t *l2, int32_t *l3, int32_t *cls, int32_t reset);
void ea_wrap(const int32_t xin[22 * 500], int8_t qout[22 * 500], int32_t reset);
int main(int, char **argv) {
    FILE *f = fopen(argv[1], "rb"); int32_t n; fread(&n, 4, 1, f);
    static int32_t X[2][11000]; static int8_t Q[2][11000]; int32_t L[2][4];
    for (int k = 0; k < 2; k++) { int32_t h[2]; fread(h, 4, 2, f); fread(X[k], 4, 11000, f); printf("trial %d session %d label %d\n", k, h[0], h[1]); }
    for (int k = 0; k < 2; k++) ea_wrap(X[k], Q[k], k == 0);
    for (int k = 0; k < 2; k++) { int32_t c; mi_bci_top(X[k], &L[k][0], &L[k][1], &L[k][2], &L[k][3], &c, k == 0); printf("%d %d %d %d %d\n", c, L[k][0], L[k][1], L[k][2], L[k][3]); }
    FILE *o = fopen("cosim_in.bin", "wb"); fwrite(X, 1, sizeof X, o); fclose(o);
    o = fopen("cosim_q.bin", "wb"); fwrite(Q, 1, sizeof Q, o); fclose(o);
    o = fopen("cosim_log.bin", "wb"); fwrite(L, 1, sizeof L, o); fclose(o);
}
