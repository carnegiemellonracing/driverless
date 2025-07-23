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
//! \date Apr 10, 2015
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

class AppBaseStatusDefinitions
{
public:
    enum class AppBaseStatusId : uint8_t
    {
        Recording = 0x01
    };
}; // AppBaseStatusDefinitions

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
