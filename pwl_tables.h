// PWL tables (Q1.15) — generated from approx.py
#pragma once
#include <stdint.h>
#define N_EXP 8
#define N_RECIP 16
// knots of 2^f in Q2.14
static const int32_t EXP_KNOTS_Q14[9] = {16384, 17867, 19484, 21247, 23170, 25268, 27554, 30048, 32768};
static const int32_t RECIP_KNOTS_Q15[17] = {32768, 30840, 29127, 27594, 26214, 24966, 23831, 22795, 21845, 20972, 20165, 19418, 18725, 18079, 17476, 16913, 16384};
