//==============================================================================
//! \file
//!
//! $$MICROVISION_LICENSE_BEGIN$$
//! Copyright (c) 2025 MicroVision, Inc., Redmond, U.S.A.
//! All Rights Reserved.
//!
//! For more details, please refer to the accompanying file
//! License.txt.
//! $$MICROVISION_LICENSE_END$$
//!
//! \date Apr 01, 2025
//------------------------------------------------------------------------------

//==============================================================================

#pragma once

//==============================================================================

#if defined(__clang__)
#    define UNUSEDLAMBDACAPTURE_WARNING_ENABLED _Pragma("GCC diagnostic error \"-Wunused-lambda-capture\"")
#    define UNUSEDLAMBDACAPTURE_WARNING_DISABLED _Pragma("GCC diagnostic ignored \"-Wunused-lambda-capture\"")

#else // no clang compiler
#    define UNUSEDLAMBDACAPTURE_WARNING_ENABLED
#    define UNUSEDLAMBDACAPTURE_WARNING_DISABLED
#endif
