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
//! \date Oct 23, 2015
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/devices/IdcTcpDevice.hpp>

#include <microvision/common/sdk/datablocks/commands/ecucommands/EcuCommand.hpp>
#include <microvision/common/sdk/datablocks/commands/ecucommands/EcuCommandC.hpp>

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
//!\brief Device class to connect to the MVIS Tracking Box.
//!\remark Since the MVIS TrackingBox is internally quite similar
//!        to a MVIS ECU, the ECU commands are used here.
//------------------------------------------------------------------------------
class MICROVISION_SDK_DEPRECATED MvisTrackingBox final : public IdcTcpDevice
{
public:
    //========================================
    //!\brief Create an MvisTrackingBox (connection class).
    //!
    //! This constructor will create an MvisTrackingBox class object
    //! which will try to connect to an TrackingBox,
    //! using the given IP address and port number.
    //!
    //!\param[in] ip    IP address of the TrackingBox
    //!                 to be connected with.
    //!\param[in] port  Port number for the connection
    //!                 with the scanner.
    //----------------------------------------
    MvisTrackingBox(const std::string& ip, const unsigned short port = 12002);

    //========================================
    //!\brief Destructor.
    //!
    //! Will disconnect before destruction.
    //----------------------------------------
    virtual ~MvisTrackingBox();

public:
    //========================================
    //!\brief Establish the connection to the hardware.
    //!
    //! Reimplements IdcDevice::getConnected. In
    //! addition it will send a setFilter command
    //! to the TrackingBox to make all messages passes its
    //! output filter.
    //!\param[in] timeoutSec set device connection timeout in sec.
    //----------------------------------------
    virtual void getConnected(const uint32_t timeoutSec = IdcEthDevice::defaultReceiveTimeoutSeconds);

public:
    //========================================
    //!\brief Send a command which expects no reply.
    //!\param[in] cmd  Command to be sent.
    //!\return The result of the operation.
    //----------------------------------------
    virtual StatusCode sendCommand(const EcuCommandCBase& cmd, const SpecialExporterBase<CommandCBase>& exporter);

    //========================================
    //!\brief Send a command and wait for a reply.
    //!
    //! The command will be sent. The calling thread
    //! will sleep until a reply has been received
    //! but not longer than the number of milliseconds
    //! given in \a timeOut.
    //!
    //!\param[in]       cmd    Command to be sent.
    //!\param[in, out]  reply  The reply container for
    //!                        the reply to be stored into.
    //!\param[in]       timeOut  Number of milliseconds to
    //!                          wait for a reply.
    //!\return The result of the operation.
    //----------------------------------------
    virtual StatusCode sendCommand(const EcuCommandCBase& cmd,
                                   const SpecialExporterBase<CommandCBase>& exporter,
                                   EcuCommandReplyBase& reply,
                                   const boost::posix_time::time_duration timeOut
                                   = boost::posix_time::milliseconds(500));

private:
    using IdcTcpDevice::sendCommand;

    //========================================
    //!\brief setFilter command to the TrackingBox to make all messages passes
    //!       its output filter
    //----------------------------------------
    void setDataTypeFilter();
}; // MvisTrackingBox

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================

// Reset compiler warning settings to default.
ALLOW_WARNINGS_END
