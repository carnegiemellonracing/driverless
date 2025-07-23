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
//! \date Oct 04, 2013
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/devices/IdcTcpDevice.hpp>
#include <microvision/common/sdk/datablocks/commands/miniluxcommands/miniluxcommands.hpp>

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
//!\brief Class to connect to a MiniLux sensor.
//!\date Oct 1, 2013
//------------------------------------------------------------------------------
class MICROVISION_SDK_DEPRECATED MvisMiniLux final : public IdcTcpDevice
{
public:
    //========================================
    //!\brief Create an MvisMiniLux (connection class).
    //!
    //! This constructor will create an MvisMiniLux class object
    //! which will try to connect to a MiniLux sensor,
    //! using the given IP address and port number.
    //!
    //! \param[in] ip    IP address of the scanner
    //!                  to be connected with.
    //! \param[in] port  Port number for the connection
    //!                  with the scanner.
    //----------------------------------------
    MvisMiniLux(const std::string& ip, const unsigned short port = 12006);

    //========================================
    //!\brief Destructor.
    //!
    //! Will disconnect before destruction.
    //----------------------------------------
    virtual ~MvisMiniLux();

public:
    //========================================
    //!\brief Send a command which expects no reply.
    //! \param[in] cmd  Command to be sent.
    //! \return The result of the operation.
    //----------------------------------------
    virtual StatusCode sendCommand(const MiniLuxCommandCBase& cmd, const SpecialExporterBase<CommandCBase>& exporter);

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
    //----------------------------------------
    virtual StatusCode sendCommand(const MiniLuxCommandCBase& cmd,
                                   const SpecialExporterBase<CommandCBase>& exporter,
                                   MiniLuxCommandReplyBase& reply,
                                   const boost::posix_time::time_duration timeOut
                                   = boost::posix_time::milliseconds(500));

private:
    using IdcTcpDevice::sendCommand;
}; // MvisMiniLux

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================

// Reset compiler warning settings to default.
ALLOW_WARNINGS_END
