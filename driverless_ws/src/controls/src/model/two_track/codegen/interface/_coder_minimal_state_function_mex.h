//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
// File: _coder_minimal_state_function_mex.h
//
// MATLAB Coder version            : 5.5
// C/C++ source code generated on  : 05-Sep-2023 12:02:37
//

#ifndef _CODER_MINIMAL_STATE_FUNCTION_MEX_H
#define _CODER_MINIMAL_STATE_FUNCTION_MEX_H

// Include Files
#include "emlrt.h"
#include "mex.h"
#include "tmwtypes.h"

// Function Declarations
MEXFUNCTION_LINKAGE void mexFunction(int32_T nlhs, mxArray *plhs[],
                                     int32_T nrhs, const mxArray *prhs[]);

emlrtCTX mexFunctionCreateRootTLS();

void unsafe_minimal_state_function_mexFunction(int32_T nlhs, mxArray *plhs[1],
                                               int32_T nrhs,
                                               const mxArray *prhs[2]);

#endif
//
// File trailer for _coder_minimal_state_function_mex.h
//
// [EOF]
//
