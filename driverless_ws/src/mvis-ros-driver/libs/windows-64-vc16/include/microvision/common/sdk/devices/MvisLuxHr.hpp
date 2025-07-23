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
//! \date Jul 30, 2015
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/devices/IdcTcpDevice.hpp>

//==============================================================================

// Change the compiler warning settings until ALLOW_WARNINGS_END.
ALLOW_WARNINGS_BEGIN
// Allow deprecated warnings in deprecated code to avoid
// compiler errors because of deprecated dependencies.
ALLOW_WARNING_DEPRECATED

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//!\brief Class to connect to a LuxHr sensor.
//!\date Jul 30, 2015
//------------------------------------------------------------------------------
class MICROVISION_SDK_DEPRECATED MvisLuxHr final : public IdcTcpDevice
{
public:
    //========================================
    //!\brief Create an MvisLuxHr (connection class).
    //!
    //! This constructor will create an MvisLuxHr class object
    //! which will try to connect to a LuxHr sensor,
    //! using the given IP address and port number.
    //!
    //! \param[in] ip    IP address of the scanner
    //!                  to be connected with.
    //! \param[in] port  Port number for the connection
    //!                  with the scanner.
    ///----------------------------------------
    MvisLuxHr(const std::string& ip, const unsigned short port = 12008);

    //========================================
    //!\brief Destructor.
    //!
    //! Will disconnect before destruction.
    ///----------------------------------------
    virtual ~MvisLuxHr();

}; // MvisLuxHr

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================

ALLOW_WARNINGS_END
