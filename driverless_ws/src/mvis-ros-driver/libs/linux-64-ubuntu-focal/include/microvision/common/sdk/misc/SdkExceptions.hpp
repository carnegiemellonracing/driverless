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
//! \date Jan 23, 2018
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>
#include <exception>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

class ContainerMismatch final : public std::exception
{
public:
    virtual const char* what() const noexcept override { return msg; }

public:
    static constexpr const char* msg{"ContainerMismatch: Dynamic container casting failed."};
}; // ContainerMismatch

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
