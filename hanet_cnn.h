// Integer-only decoder network (deployed form of Section III-D/F):
//   in int8 (22 x 500, EA-aligned)  -> spatial mix (16x22)      -> requant int8   "spat"
//   -> depthwise temporal conv K1=64 (+bias) -> requant Q.8 -> HW-ELU -> avg-pool 4 -> int8 "blk1"
//   -> depthwise conv K2=16 -> requant int8 "dw"
//   -> pointwise 16x16 (+bias) -> requant Q.8 -> HW-ELU -> avg-pool 8 -> int8 "tok" (16 x 15)
//   -> linear 240 -> 4 : int32 logits (real logit = acc * LOGIT_SCALE[n])
// Bit-accurate to python/deploy.py::forward_deploy up to rounding of the requant multipliers.
#pragma once
#include <stdint.h>

#define CNN_C 22
#define CNN_T 500
#define CNN_F2 16
#define CNN_K1 64
#define CNN_K2 16
#define CNN_P1 4
#define CNN_P2 8
#define CNN_T1 (CNN_T / CNN_P1)   // 125
#define CNN_L (CNN_T1 / CNN_P2)   // 15
#define CNN_NCLS 4

void hanet_cnn(const int8_t x[CNN_C][CNN_T], int32_t logits[CNN_NCLS]);

// shared helpers (also used by the test bench)
int8_t requant8(int64_t acc, int32_t M, int sh);
int32_t elu_q8(int32_t v8, int32_t k16, int32_t inv16);   // input/output: Q.8 in units of s
