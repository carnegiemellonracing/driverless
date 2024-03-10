//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
// File: _coder_minimal_state_function_api.h
//
// MATLAB Coder version            : 5.5
// C/C++ source code generated on  : 05-Sep-2023 12:02:37
//

#ifndef _CODER_MINIMAL_STATE_FUNCTION_API_H
#define _CODER_MINIMAL_STATE_FUNCTION_API_H

// Include Files
#include "emlrt.h"
#include "tmwtypes.h"
#include <algorithm>
#include <cstring>

// Variable Declarations
extern emlrtCTX emlrtRootTLSGlobal;
extern emlrtContext emlrtContextGlobal;

// Function Declarations
void minimal_state_function(real_T in1[10], real_T in2[5], real_T out1[10]);

void minimal_state_function_api(const mxArray *const prhs[2],
                                const mxArray **plhs);

void minimal_state_function_atexit();

void minimal_state_function_initialize();

void minimal_state_function_terminate();

void minimal_state_function_xil_shutdown();

void minimal_state_function_xil_terminate();

#endif
//
// File trailer for _coder_minimal_state_function_api.h
//
// [EOF]
//
