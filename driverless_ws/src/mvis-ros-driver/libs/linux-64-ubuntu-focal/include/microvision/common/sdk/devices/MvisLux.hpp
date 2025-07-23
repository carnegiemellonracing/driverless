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
//! \date Apr 24, 2012
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/devices/IdcTcpDevice.hpp>

#include <microvision/common/sdk/datablocks/commands/luxcommands/LuxCommand.hpp>
#include <microvision/common/sdk/datablocks/commands/luxcommands/LuxCommandC.hpp>

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
//!\brief Class to connect to a LUX3/LUX4 sensor.
//!\date Oct 1, 2013
//------------------------------------------------------------------------------
class MICROVISION_SDK_DEPRECATED MvisLux final : public IdcTcpDevice
{
public:
    //========================================
    //!\brief Create an MvisLux (connection class).
    //!
    //! This constructor will create an MvisLux class object
    //! which will try to connect to a LUX3/LUX4 sensor,
    //! using the given IP address and port number.
    //!
    //! \param[in] ip    IP address of the scanner
    //!                  to be connected with.
    //! \param[in] port  Port number for the connection
    //!                  with the scanner.
    ///----------------------------------------
    MvisLux(const std::string& ip, const unsigned short port = 12002);

    //========================================
    //!\brief Destructor.
    //!
    //! Will disconnect before destruction.
    ///----------------------------------------
    virtual ~MvisLux();

public:
    //========================================
    //!\brief Send a command which expects no reply.
    //! \param[in] cmd  Command to be sent.
    //! \return The result of the operation.
    ///----------------------------------------
    virtual StatusCode sendCommand(const LuxCommandCBase& cmd, const SpecialExporterBase<CommandCBase>& exporter);

    //========================================
    //!\brief Send a command and wait for a reply.
    //!
    //! The command will be sent. The calling thread
    //! will sleep until a reply has been received
    //! but not longer than the number of milliseconds
    //! given in \a timeOut.
    //!
    //! \param[in]       cmd    Command to be sent.
    //! \param[in, out]  reply  The reply container for
    //!                         the reply to be stored into.
    //! \param[in]       timeOut  Number of milliseconds to
    //!                           wait for a reply.
    //! \return The result of the operation.
    ///----------------------------------------
    virtual StatusCode sendCommand(const LuxCommandCBase& cmd,
                                   const SpecialExporterBase<CommandCBase>& exporter,
                                   LuxCommandReplyBase& reply,
                                   const boost::posix_time::time_duration timeOut
                                   = boost::posix_time::milliseconds(500));

private:
    using IdcTcpDevice::sendCommand;
}; // MvisLux

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================

ALLOW_WARNINGS_END
