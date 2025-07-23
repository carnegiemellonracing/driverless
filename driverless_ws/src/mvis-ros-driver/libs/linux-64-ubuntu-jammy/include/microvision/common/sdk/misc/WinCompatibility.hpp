//==============================================================================
//! \file
//!
//!        Compatibility file for windows (Visual Studio).
//!
//!\note: This file is a basic include. It should be included
//!       by all header files.
//!
//! \note: This file has an embedded auto include for the stdInt data type:
//!        - <stdint.h> (for linux and Visual Studio newer than 2008);
//!        - <stdintForVS2008.h> for VS 2008.
//!
//! $$MICROVISION_LICENSE_BEGIN$$
//! Copyright (c) 2025 MicroVision, Inc., Redmond, U.S.A.
//! All Rights Reserved.
//!
//! For more details, please refer to the accompanying file
//! License.txt.
//! $$MICROVISION_LICENSE_END$$
//!
//! \date Oct 2, 2013
//------------------------------------------------------------------------------

#pragma once

//==============================================================================
//! \brief Deprecated.
//! Please use the new define includes in mvis/common/sdk/misc/defines/*.
//------------------------------------------------------------------------------
#include <microvision/common/sdk/misc/defines/windows.hpp>
#include <microvision/common/sdk/misc/defines/ioAbstractions.hpp>
#include <microvision/common/sdk/misc/defines/general.hpp>