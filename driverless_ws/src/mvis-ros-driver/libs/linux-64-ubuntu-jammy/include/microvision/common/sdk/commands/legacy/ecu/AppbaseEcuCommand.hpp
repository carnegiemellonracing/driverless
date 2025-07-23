//==============================================================================
//! \file
//!
//! \brief Command for Appbase ECU device.
//!
//! $$MICROVISION_LICENSE_BEGIN$$
//! Copyright (c) 2025 MicroVision, Inc., Redmond, U.S.A.
//! All Rights Reserved.
//!
//! For more details, please refer to the accompanying file
//! License.txt.
//! $$MICROVISION_LICENSE_END$$
//!
//! \date Dec 10, 2021
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/commands/Command.hpp>
#include <microvision/common/sdk/io/IdcDataPackage.hpp>

#include <tuple>
#include <future>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Command for Appbase ECU device.
//!
//! \extends Command
//------------------------------------------------------------------------------
class AppbaseEcuCommand : public Command
{
public:
    //========================================
    //! \brief Data to send command with reply callback.
    //----------------------------------------
    struct SendFunctionParameters
    {
        //========================================
        //! \brief Command request data.
        //----------------------------------------
        DataPackagePtr data;

        //========================================
        //! \brief Command reply callback.
        //----------------------------------------
        std::function<void(const IdcDataPackagePtr&)> replyFunction;
    };

    //========================================
    //! \brief Pointer type of SendFunctionParameters.
    //----------------------------------------
    using SendFunctionParametersPtr = std::shared_ptr<SendFunctionParameters>;

    //========================================
    //! \brief Function to send data via ECU device.
    //----------------------------------------
    using SendFunctionType = std::function<void(const SendFunctionParametersPtr&)>;

private:
    //========================================
    //! \brief Logger
    //----------------------------------------
    static MICROVISION_SDK_API logging::LoggerSPtr m_logger;

public:
    //========================================
    //! \brief Constructor
    //! \param[in]  originator  The originator of this command.
    //! \param[in]  request     Request which shall be used for this command.
    //! \param[in]  reply       Reply which shall be used for this command.
    //! \param[in]  sendCommand Function to send command via ECU device.
    //--------------------------------------
    AppbaseEcuCommand(CommandInterfaceTagUPtr&& originator,
                      ConfigurationUPtr&& request,
                      ConfigurationUPtr&& reply,
                      const SendFunctionType& sendCommand);

    //========================================
    //! \brief Copy constructor (deleted)
    //----------------------------------------
    AppbaseEcuCommand(const AppbaseEcuCommand&) = delete;

    //========================================
    //! \brief Move constructor (deleted)
    //----------------------------------------
    AppbaseEcuCommand(AppbaseEcuCommand&&) = delete;

    //========================================
    //! \brief Destructor.
    //----------------------------------------
    ~AppbaseEcuCommand() override;

public:
    //========================================
    //! \brief Get progress status of command execution.
    //! \param[in] waitTimeInMs  Will not be used at all.
    //! \returns  Command execution progress status.
    //! \note The progress state is returned immediately and the thread will not be blocked at all.
    //----------------------------------------
    StatusCode getExecutionProgressStatus(const uint32_t waitTimeInMs) const override;

    //========================================
    //! \brief Get error status of command execution.
    //! \returns  Command error status.
    //----------------------------------------
    StatusCode getExecutionErrorStatus() const override;

protected:
    //========================================
    //! \brief Internal implementation of command execution.
    //----------------------------------------
    void executeInternal() override;

    //========================================
    //! \brief Internal implementation of command abortion.
    //----------------------------------------
    void abortInternal() override;

private:
    //========================================
    //! \brief Function to send data via ECU device.
    //----------------------------------------
    SendFunctionType m_sendFunction;

    //========================================
    //! \brief Result of the last command execution.
    //----------------------------------------
    StatusCode m_progressStatus;

    //========================================
    //! \brief Error status;
    //----------------------------------------
    StatusCode m_errorStatus;
}; // class AppbaseEcuCommand

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
