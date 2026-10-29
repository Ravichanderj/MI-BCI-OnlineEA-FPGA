// Testbench: integer base-2 softmax vs exact float softmax on random attention rows.
// g++ -O2 -std=c++14 tb_softmax.cpp -o tb_softmax && ./tb_softmax
#include <cmath>
#include <cstdio>
#include <random>
#include "softmax_pow2.h"

int main() {
    const int L = 15;                 // tokens in HA-Net (500 / 32)
    std::mt19937 g(0);
    std::normal_distribution<double> nd(0.0, 3.0);
    double max_err = 0, mean_err = 0, max_sum_err = 0;
    int argmax_ok = 0, rows = 20000;
    for (int r = 0; r < rows; r++) {
        double z[L]; int16_t s[L]; uint16_t p[L];
        for (int j = 0; j < L; j++) {
            z[j] = nd(g);                                         // natural-log-domain score
            double u = z[j] * 1.4426950408889634;                  // log2(e) folded in W_q
            s[j] = (int16_t)std::lround(u * 256.0);                // Q7.8
        }
        softmax_pow2_row<L>(s, p);
        double mx = z[0]; for (int j = 1; j < L; j++) mx = std::fmax(mx, z[j]);
        double den = 0, ref[L]; for (int j = 0; j < L; j++) den += ref[j] = std::exp(z[j] - mx);
        double sum = 0; int a1 = 0, a2 = 0;
        for (int j = 0; j < L; j++) {
            ref[j] /= den;
            double pj = p[j] / 32768.0;
            double e = std::fabs(pj - ref[j]);
            max_err = std::fmax(max_err, e); mean_err += e; sum += pj;
            if (ref[j] > ref[a1]) a1 = j;
            if (p[j] > p[a2]) a2 = j;
        }
        max_sum_err = std::fmax(max_sum_err, std::fabs(sum - 1));
        argmax_ok += (a1 == a2);
    }
    printf("rows=%d | max |p-p_ref| = %.2e | mean = %.2e | max |sum-1| = %.2e | argmax agree %.2f%%\n",
           rows, max_err, mean_err / rows / L, max_sum_err, 100.0 * argmax_ok / rows);
    bool pass = max_err < 5e-3;
    printf("%s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
