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
//! \date Jan 20, 2014
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <sstream>
#include <string>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

inline uint16_t getPort(std::string& ip, const uint16_t defaultPort = 0)
{
    const size_t portPos = ip.find(':');
    uint16_t port        = defaultPort;
    if (portPos != std::string::npos)
    {
        std::string portStr = ip.substr(portPos + 1);
        std::stringstream ss;
        ss.str(portStr);
        ss >> port;
        ip = ip.substr(0, portPos);
    }

    return port;
}

//==============================================================================

} // namespace sdk
} // namespace common
} // namespace microvision

//==============================================================================
