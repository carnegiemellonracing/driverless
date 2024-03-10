//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
// File: _coder_minimal_state_function_api.cpp
//
// MATLAB Coder version            : 5.5
// C/C++ source code generated on  : 05-Sep-2023 12:02:37
//

// Include Files
#include "_coder_minimal_state_function_api.h"
#include "_coder_minimal_state_function_mex.h"

// Variable Definitions
emlrtCTX emlrtRootTLSGlobal{nullptr};

emlrtContext emlrtContextGlobal{
    true,                                                 // bFirstTime
    false,                                                // bInitialized
    131627U,                                              // fVersionInfo
    nullptr,                                              // fErrorFunction
    "minimal_state_function",                             // fFunctionName
    nullptr,                                              // fRTCallStack
    false,                                                // bDebugMode
    {2045744189U, 2170104910U, 2743257031U, 4284093946U}, // fSigWrd
    nullptr                                               // fSigMem
};

// Function Declarations
static real_T (*b_emlrt_marshallIn(const emlrtStack *sp, const mxArray *in2,
                                   const char_T *identifier))[5];

static real_T (*b_emlrt_marshallIn(const emlrtStack *sp, const mxArray *u,
                                   const emlrtMsgIdentifier *parentId))[5];

static real_T (*c_emlrt_marshallIn(const emlrtStack *sp, const mxArray *src,
                                   const emlrtMsgIdentifier *msgId))[10];

static real_T (*d_emlrt_marshallIn(const emlrtStack *sp, const mxArray *src,
                                   const emlrtMsgIdentifier *msgId))[5];

static real_T (*emlrt_marshallIn(const emlrtStack *sp, const mxArray *in1,
                                 const char_T *identifier))[10];

static real_T (*emlrt_marshallIn(const emlrtStack *sp, const mxArray *u,
                                 const emlrtMsgIdentifier *parentId))[10];

static const mxArray *emlrt_marshallOut(const real_T u[10]);

// Function Definitions
//
// Arguments    : const emlrtStack *sp
//                const mxArray *in2
//                const char_T *identifier
// Return Type  : real_T (*)[5]
//
static real_T (*b_emlrt_marshallIn(const emlrtStack *sp, const mxArray *in2,
                                   const char_T *identifier))[5]
{
  emlrtMsgIdentifier thisId;
  real_T(*y)[5];
  thisId.fIdentifier = const_cast<const char_T *>(identifier);
  thisId.fParent = nullptr;
  thisId.bParentIsCell = false;
  y = b_emlrt_marshallIn(sp, emlrtAlias(in2), &thisId);
  emlrtDestroyArray(&in2);
  return y;
}

//
// Arguments    : const emlrtStack *sp
//                const mxArray *u
//                const emlrtMsgIdentifier *parentId
// Return Type  : real_T (*)[5]
//
static real_T (*b_emlrt_marshallIn(const emlrtStack *sp, const mxArray *u,
                                   const emlrtMsgIdentifier *parentId))[5]
{
  real_T(*y)[5];
  y = d_emlrt_marshallIn(sp, emlrtAlias(u), parentId);
  emlrtDestroyArray(&u);
  return y;
}

//
// Arguments    : const emlrtStack *sp
//                const mxArray *src
//                const emlrtMsgIdentifier *msgId
// Return Type  : real_T (*)[10]
//
static real_T (*c_emlrt_marshallIn(const emlrtStack *sp, const mxArray *src,
                                   const emlrtMsgIdentifier *msgId))[10]
{
  static const int32_T dims{10};
  real_T(*ret)[10];
  emlrtCheckBuiltInR2012b((emlrtConstCTX)sp, msgId, src, "double", false, 1U,
                          (const void *)&dims);
  ret = (real_T(*)[10])emlrtMxGetData(src);
  emlrtDestroyArray(&src);
  return ret;
}

//
// Arguments    : const emlrtStack *sp
//                const mxArray *src
//                const emlrtMsgIdentifier *msgId
// Return Type  : real_T (*)[5]
//
static real_T (*d_emlrt_marshallIn(const emlrtStack *sp, const mxArray *src,
                                   const emlrtMsgIdentifier *msgId))[5]
{
  static const int32_T dims{5};
  real_T(*ret)[5];
  emlrtCheckBuiltInR2012b((emlrtConstCTX)sp, msgId, src, "double", false, 1U,
                          (const void *)&dims);
  ret = (real_T(*)[5])emlrtMxGetData(src);
  emlrtDestroyArray(&src);
  return ret;
}

//
// Arguments    : const emlrtStack *sp
//                const mxArray *in1
//                const char_T *identifier
// Return Type  : real_T (*)[10]
//
static real_T (*emlrt_marshallIn(const emlrtStack *sp, const mxArray *in1,
                                 const char_T *identifier))[10]
{
  emlrtMsgIdentifier thisId;
  real_T(*y)[10];
  thisId.fIdentifier = const_cast<const char_T *>(identifier);
  thisId.fParent = nullptr;
  thisId.bParentIsCell = false;
  y = emlrt_marshallIn(sp, emlrtAlias(in1), &thisId);
  emlrtDestroyArray(&in1);
  return y;
}

//
// Arguments    : const emlrtStack *sp
//                const mxArray *u
//                const emlrtMsgIdentifier *parentId
// Return Type  : real_T (*)[10]
//
static real_T (*emlrt_marshallIn(const emlrtStack *sp, const mxArray *u,
                                 const emlrtMsgIdentifier *parentId))[10]
{
  real_T(*y)[10];
  y = c_emlrt_marshallIn(sp, emlrtAlias(u), parentId);
  emlrtDestroyArray(&u);
  return y;
}

//
// Arguments    : const real_T u[10]
// Return Type  : const mxArray *
//
static const mxArray *emlrt_marshallOut(const real_T u[10])
{
  static const int32_T i{0};
  static const int32_T i1{10};
  const mxArray *m;
  const mxArray *y;
  y = nullptr;
  m = emlrtCreateNumericArray(1, (const void *)&i, mxDOUBLE_CLASS, mxREAL);
  emlrtMxSetData((mxArray *)m, (void *)&u[0]);
  emlrtSetDimensions((mxArray *)m, &i1, 1);
  emlrtAssign(&y, m);
  return y;
}

//
// Arguments    : const mxArray * const prhs[2]
//                const mxArray **plhs
// Return Type  : void
//
void minimal_state_function_api(const mxArray *const prhs[2],
                                const mxArray **plhs)
{
  emlrtStack st{
      nullptr, // site
      nullptr, // tls
      nullptr  // prev
  };
  real_T(*in1)[10];
  real_T(*out1)[10];
  real_T(*in2)[5];
  st.tls = emlrtRootTLSGlobal;
  out1 = (real_T(*)[10])mxMalloc(sizeof(real_T[10]));
  // Marshall function inputs
  in1 = emlrt_marshallIn(&st, emlrtAlias(prhs[0]), "in1");
  in2 = b_emlrt_marshallIn(&st, emlrtAlias(prhs[1]), "in2");
  // Invoke the target function
  minimal_state_function(*in1, *in2, *out1);
  // Marshall function outputs
  *plhs = emlrt_marshallOut(*out1);
}

//
// Arguments    : void
// Return Type  : void
//
void minimal_state_function_atexit()
{
  emlrtStack st{
      nullptr, // site
      nullptr, // tls
      nullptr  // prev
  };
  mexFunctionCreateRootTLS();
  st.tls = emlrtRootTLSGlobal;
  emlrtEnterRtStackR2012b(&st);
  emlrtDestroyRootTLS(&emlrtRootTLSGlobal);
  minimal_state_function_xil_terminate();
  minimal_state_function_xil_shutdown();
  emlrtExitTimeCleanup(&emlrtContextGlobal);
}

//
// Arguments    : void
// Return Type  : void
//
void minimal_state_function_initialize()
{
  emlrtStack st{
      nullptr, // site
      nullptr, // tls
      nullptr  // prev
  };
  mexFunctionCreateRootTLS();
  st.tls = emlrtRootTLSGlobal;
  emlrtClearAllocCountR2012b(&st, false, 0U, nullptr);
  emlrtEnterRtStackR2012b(&st);
  emlrtFirstTimeR2012b(emlrtRootTLSGlobal);
}

//
// Arguments    : void
// Return Type  : void
//
void minimal_state_function_terminate()
{
  emlrtDestroyRootTLS(&emlrtRootTLSGlobal);
}

//
// File trailer for _coder_minimal_state_function_api.cpp
//
// [EOF]
//
