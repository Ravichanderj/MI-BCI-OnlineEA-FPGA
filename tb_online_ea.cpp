// C-simulation testbench: python reference vs double core vs fixed-point core vs HW top.
// Build outside Vitis:  g++ -O2 -std=c++14 -I<ap_types>/include tb_online_ea.cpp online_ea.cpp
#include <cmath>
#include <cstdio>
#include <vector>
#include "online_ea.h"

void online_ea_ref(const double x[EA_C][EA_T], double W[EA_C][EA_C], bool reset);
void online_ea_fixed_W(const din_t x[EA_C][EA_T], double W[EA_C][EA_C], bool reset);

static double rel_err(const double A[EA_C][EA_C], const double B[EA_C][EA_C]) {
    double num = 0, den = 0;
    for (int i = 0; i < EA_C; i++) for (int j = 0; j < EA_C; j++) {
        num += (A[i][j] - B[i][j]) * (A[i][j] - B[i][j]); den += B[i][j] * B[i][j];
    }
    return std::sqrt(num / den);
}

int main() {
    FILE *f = fopen("ea_in.bin", "rb");
    if (!f) { printf("run: python3 make_ea_testdata.py first\n"); return 1; }
    fseek(f, 0, SEEK_END); long bytes = ftell(f); fseek(f, 0, SEEK_SET);
    int n = bytes / (sizeof(float) * EA_C * EA_T);
    std::vector<float> buf(bytes / sizeof(float));
    fread(buf.data(), sizeof(float), buf.size(), f); fclose(f);
    double Wpy[EA_C][EA_C];
    f = fopen("ea_ref_W.bin", "rb"); fread(Wpy, sizeof(double), EA_C * EA_C, f); fclose(f);

    static double xd[EA_C][EA_T]; static din_t xf[EA_C][EA_T]; static dout_t y[EA_C][EA_T];
    double Wd[EA_C][EA_C], Wf[EA_C][EA_C];
    for (int k = 0; k < n; k++) {
        for (int i = 0; i < EA_C; i++) for (int t = 0; t < EA_T; t++) {
            float v = buf[(size_t)k * EA_C * EA_T + i * EA_T + t];
            xd[i][t] = v; xf[i][t] = (din_t)v;
        }
        online_ea_ref(xd, Wd, k == 0);
        online_ea_fixed_W(xf, Wf, k == 0);
        online_ea(xf, y, k == 0);
    }
    double e_ref = rel_err(Wd, Wpy), e_fx = rel_err(Wf, Wpy);
    // aligned output of the last trial should have ~unit variance per channel
    double var = 0;
    for (int i = 0; i < EA_C; i++) for (int t = 0; t < EA_T; t++) var += std::pow(y[i][t].to_double(), 2);
    var /= EA_C * EA_T;
    printf("trials=%d | rel.err W: double-core vs python %.2e | fixed-point vs python %.2e\n", n, e_ref, e_fx);
    printf("HW top: mean output variance of last trial = %.3f (expect ~1)\n", var);
    bool pass = e_ref < 1e-5 && e_fx < 1e-2;
    printf("%s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
