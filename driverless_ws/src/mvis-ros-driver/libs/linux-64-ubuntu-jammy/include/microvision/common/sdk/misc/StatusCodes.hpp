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
//! \date Apr 14, 2015
//------------------------------------------------------------------------------

#pragma once

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <iostream>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

enum class StatusCode : uint8_t
{
    EverythingOk = 0, //!< No error
    NotConnected, //!< Device is not connected.
    MismatchingCommandAndReply, //!<
    FailedToPrepareSendingCommand, //!< Error in preparing the send buffer.
    FailedToPrepareSendingDataBlock, //!< Error in preparing the send buffer.
    SendingCommandFailed, //!< Sending the command has failed.
    SendingDataBlockFailed, //!< Sending of the data block has failed.
    ReplyMismatch, //!< Received wrong command reply. Command ids not matching.
    TimeOut, //!< The reply has not received in proper time.
    TimeOutCriticalSection, //!< Critical section was blocked too long.
    ReceiveCommandErrorReply, //!< An command error reply has been received.
    DataBlockBlockedByFilter, //!< Datablock blocked by datatype id filter
    StreamerNotRegistered,
    ListenerNotRegistered,
    NotStartedYet,
    InProgress,
    Finished
}; // StatusCode

//==============================================================================

std::ostream& operator<<(std::ostream& os, const StatusCode ec);

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
