//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
// File: main.cpp
//
// MATLAB Coder version            : 5.5
// C/C++ source code generated on  : 05-Sep-2023 12:02:37
//

/*************************************************************************/
/* This automatically generated example C++ main file shows how to call  */
/* entry-point functions that MATLAB Coder generated. You must customize */
/* this file for your application. Do not modify this file directly.     */
/* Instead, make a copy of this file, modify it, and integrate it into   */
/* your development environment.                                         */
/*                                                                       */
/* This file initializes entry-point function arguments to a default     */
/* size and value before calling the entry-point functions. It does      */
/* not store or use any values returned from the entry-point functions.  */
/* If necessary, it does pre-allocate memory for returned values.        */
/* You can use this file as a starting point for a main function that    */
/* you can deploy in your application.                                   */
/*                                                                       */
/* After you copy the file, and before you deploy it, you must make the  */
/* following changes:                                                    */
/* * For variable-size function arguments, change the example sizes to   */
/* the sizes that your application requires.                             */
/* * Change the example values of function arguments to the values that  */
/* your application requires.                                            */
/* * If the entry-point functions return values, store these values or   */
/* otherwise use them as required by your application.                   */
/*                                                                       */
/*************************************************************************/

// Include Files
#include "main.h"
#include "minimal_state_function.h"
#include "minimal_state_function_terminate.h"
#include "rt_nonfinite.h"

// Function Declarations
static void argInit_10x1_real_T(double result[10]);

static void argInit_5x1_real_T(double result[5]);

static double argInit_real_T();

// Function Definitions
//
// Arguments    : double result[10]
// Return Type  : void
//
static void argInit_10x1_real_T(double result[10])
{
  // Loop over the array to initialize each element.
  for (int idx0{0}; idx0 < 10; idx0++) {
    // Set the value of the array element.
    // Change this value to the value that the application requires.
    result[idx0] = argInit_real_T();
  }
}

//
// Arguments    : double result[5]
// Return Type  : void
//
static void argInit_5x1_real_T(double result[5])
{
  // Loop over the array to initialize each element.
  for (int idx0{0}; idx0 < 5; idx0++) {
    // Set the value of the array element.
    // Change this value to the value that the application requires.
    result[idx0] = argInit_real_T();
  }
}

//
// Arguments    : void
// Return Type  : double
//
static double argInit_real_T()
{
  return 0.0;
}

//
// Arguments    : int argc
//                char **argv
// Return Type  : int
//
int main(int, char **)
{
  // The initialize function is being called automatically from your entry-point
  // function. So, a call to initialize is not included here. Invoke the
  // entry-point functions.
  // You can call entry-point functions multiple times.
  main_minimal_state_function();
  // Terminate the application.
  // You do not need to do this more than one time.
  minimal_state_function_terminate();
  return 0;
}

//
// Arguments    : void
// Return Type  : void
//
void main_minimal_state_function()
{
  double dv[10];
  double out1[10];
  double dv1[5];
  // Initialize function 'minimal_state_function' input arguments.
  // Initialize function input argument 'in1'.
  // Initialize function input argument 'in2'.
  // Call the entry-point 'minimal_state_function'.
  argInit_10x1_real_T(dv);
  argInit_5x1_real_T(dv1);
  minimal_state_function(dv, dv1, out1);
}

//
// File trailer for main.cpp
//
// [EOF]
//
