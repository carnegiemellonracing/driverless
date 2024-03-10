//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
// File: _coder_minimal_state_function_mex.cpp
//
// MATLAB Coder version            : 5.5
// C/C++ source code generated on  : 05-Sep-2023 12:02:37
//

// Include Files
#include "_coder_minimal_state_function_mex.h"
#include "_coder_minimal_state_function_api.h"

// Function Definitions
//
// Arguments    : int32_T nlhs
//                mxArray *plhs[]
//                int32_T nrhs
//                const mxArray *prhs[]
// Return Type  : void
//
void mexFunction(int32_T nlhs, mxArray *plhs[], int32_T nrhs,
                 const mxArray *prhs[])
{
  mexAtExit(&minimal_state_function_atexit);
  // Module initialization.
  minimal_state_function_initialize();
  // Dispatch the entry-point.
  unsafe_minimal_state_function_mexFunction(nlhs, plhs, nrhs, prhs);
  // Module termination.
  minimal_state_function_terminate();
}

//
// Arguments    : void
// Return Type  : emlrtCTX
//
emlrtCTX mexFunctionCreateRootTLS()
{
  emlrtCreateRootTLSR2022a(&emlrtRootTLSGlobal, &emlrtContextGlobal, nullptr, 1,
                           nullptr, "UTF-8", true);
  return emlrtRootTLSGlobal;
}

//
// Arguments    : int32_T nlhs
//                mxArray *plhs[1]
//                int32_T nrhs
//                const mxArray *prhs[2]
// Return Type  : void
//
void unsafe_minimal_state_function_mexFunction(int32_T nlhs, mxArray *plhs[1],
                                               int32_T nrhs,
                                               const mxArray *prhs[2])
{
  emlrtStack st{
      nullptr, // site
      nullptr, // tls
      nullptr  // prev
  };
  const mxArray *outputs;
  st.tls = emlrtRootTLSGlobal;
  // Check for proper number of arguments.
  if (nrhs != 2) {
    emlrtErrMsgIdAndTxt(&st, "EMLRT:runTime:WrongNumberOfInputs", 5, 12, 2, 4,
                        22, "minimal_state_function");
  }
  if (nlhs > 1) {
    emlrtErrMsgIdAndTxt(&st, "EMLRT:runTime:TooManyOutputArguments", 3, 4, 22,
                        "minimal_state_function");
  }
  // Call the function.
  minimal_state_function_api(prhs, &outputs);
  // Copy over outputs to the caller.
  emlrtReturnArrays(1, &plhs[0], &outputs);
}

//
// File trailer for _coder_minimal_state_function_mex.cpp
//
// [EOF]
//
