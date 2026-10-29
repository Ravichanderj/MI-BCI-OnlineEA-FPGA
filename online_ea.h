// Online Euclidean Alignment — HLS kernel (Vivado HLS 2019.2; Zynq-7020 / Zynq-7010)
// Algorithm and operation order identical to python/ea.py::OnlineEA.
// v2: no floating point anywhere (scalar inverse sqrt by leading-one + fixed-point Newton),
//     resource-shared datapaths (DSP budget ~40 instead of 226).
#pragma once
#include <ap_fixed.h>
#include <ap_int.h>
#include <stdint.h>

#define EA_C 22          // channels
#define EA_T 500         // samples per trial (4 s @ 125 Hz)
#define EA_KEMA 5        // steady-state forgetting 2^-5
#define EA_NS_ITERS 18   // coupled Newton-Schulz iterations (fixed latency)
#define EA_EPS 0.01      // shrinkage
// NOTE: Vivado HLS 2019.2 does not expand macros inside #pragma HLS -> literals are used there.
//       Newton-Schulz matmul unroll = 2 (cyclic partition 2) in online_ea.cpp.

// ---- fixed-point formats (verified by tb_online_ea.cpp / tb_mi_bci_top.cpp)
typedef ap_fixed<24, 14> din_t;    // filtered EEG in uV (+-8192 uV, 2^-10 resolution)
typedef ap_fixed<24, 8>  dout_t;   // aligned EEG (unit variance)
typedef ap_fixed<48, 24> cov_t;    // covariance / running reference (uV^2)
typedef ap_fixed<32, 8>  mat_t;    // A, Y, Z, T in Newton-Schulz (|Z| <= ~50 with eps=0.01)
typedef ap_fixed<56, 16> acc_t;    // Newton-Schulz matmul accumulator, output gain
typedef ap_fixed<64, 38> cacc_t;   // covariance accumulator: sum of T=500 products of uV^2
typedef ap_fixed<64, 30> app_t;    // whitening accumulator Z*x (uV-scale, artefact-safe)
typedef ap_ufixed<34, 2> nr_t;     // mantissa inverse sqrt r_m in (0.70, 1]

#define EA_COMP 0.98821176880261852   // sqrt(500/512): HW divides the covariance sum by 512

// r = r_m * 2^-h  (fixed) or r_m (double reference, h = 0)
template <typename T> struct RScale { T rm; int h; };

void online_ea(const din_t x[EA_C][EA_T], dout_t y[EA_C][EA_T], bool reset);
void online_ea_q8(const din_t x[EA_C][EA_T], int8_t q[EA_C][EA_T], acc_t inv_s_in, bool reset);
void online_ea_q8_ref(const double x[EA_C][EA_T], int8_t q[EA_C][EA_T], double inv_s_in, bool reset);
