// Auto-generated from hanet_weights.h comments: constants for the integer ELU and logits
#pragma once
#include <stdint.h>
#define ELU_K16_BLK1 4566   // s*log2(e) in Q16
#define ELU_INV16_BLK1 1357160  // 1/s in Q16
#define ELU_K16_TOK 1530   // s*log2(e) in Q16
#define ELU_INV16_TOK 4050127  // 1/s in Q16
static const float LOGIT_SCALE[4] = {5.439000e-05f, 4.939000e-05f, 6.102000e-05f, 4.120000e-05f};
#define S_IN 3.693221e-02f   // decoder input scale (aligned EEG -> int8)
static const int32_t LOGIT_W[4] = {29208, 26523, 32768, 22125};   // relative logit scales, Q15 (argmax only)
