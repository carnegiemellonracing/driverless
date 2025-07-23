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
//! \date Jun 25, 2019
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/misc/StatusCodes.hpp>
#include <microvision/common/sdk/datablocks/commands/Command.hpp>
#include <microvision/common/sdk/datablocks/commands/Command2010.hpp>
#include <microvision/common/sdk/datablocks/SpecialExporterBase.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//=================================================
//! \brief Command interface for idc devices.
//! \sa MvisEcuDevice
//! \sa MvisTrackingBox
//! \sa MvisLuxSensorDevice
//! \sa MvisMiniLuxSensorDevice
//-------------------------------------------------
class MICROVISION_SDK_DEPRECATED Commander
{
public:
    //========================================
    //! \brief Command pointer type.
    //----------------------------------------
    using CommandPtr = std::shared_ptr<CommandCBase>;

    //========================================
    //! \brief Reply pointer type.
    //----------------------------------------
    using ReplyPtr = std::shared_ptr<CommandReplyBase>;

    //========================================
    //! \brief Command exporter type.
    //----------------------------------------
    using ExporterPtr = std::shared_ptr<SpecialExporterBase<CommandCBase>>;

    //========================================
    //! \brief Callback of command sending function type.
    //----------------------------------------
    using CallbackType = std::function<void(const StatusCode, const ReplyPtr)>;

public:
    //========================================
    //! \brief Send a command which expects no reply.
    //! \param[in] command      Command to be sent.
    //! \param[in] exporter     Exporter to serialize command.
    //! \param[in] callback     (Optional) Will called with result of command sending.
    //!                         Per default nullptr.
    //----------------------------------------
    virtual void
    sendCommand(const CommandPtr& command, const ExporterPtr& exporter, const CallbackType& callback = nullptr)
        = 0;

    //========================================
    //! \brief Send a command which expects no reply.
    //! \param[in]      command         Command to be sent.
    //! \param[in]      exporter        Exporter to serialize command.
    //! \param[in, out] reply           The reply container for the reply to be stored into.
    //! \param[in]      timeoutInMs     (Optional) Number of milliseconds to wait for a reply.
    //! \param[in]      callback        (Optional) Will called with result of command sending.
    //!                                 Per default nullptr.
    //----------------------------------------
    virtual void sendCommand(const CommandPtr& command,
                             const ExporterPtr& exporter,
                             ReplyPtr& reply,
                             const uint32_t timeoutInMs   = 500U,
                             const CallbackType& callback = nullptr)
        = 0;
};

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
