//==============================================================================
//! \file
//!
//! \brief Commands for devices.
//!
//! $$MICROVISION_LICENSE_BEGIN$$
//! Copyright (c) 2025 MicroVision, Inc., Redmond, U.S.A.
//! All Rights Reserved.
//!
//! For more details, please refer to the accompanying file
//! License.txt.
//! $$MICROVISION_LICENSE_END$$
//!
//! \date Sep 22, 2021
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/commands/CommandInterfaceTag.hpp>
#include <microvision/common/sdk/config/Configuration.hpp>
#include <microvision/common/sdk/misc/ThreadSafe.hpp>
#include <microvision/common/sdk/misc/StatusCodes.hpp>

#include <chrono>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

class CommandInterfaceTag;

//==============================================================================
//! \brief Represents a command which can be sent to devices.
//! It contains a request and response instance.
//------------------------------------------------------------------------------
class Command
{
public:
    //==============================================================================
    //! \brief Function type used for command callbacks.
    //------------------------------------------------------------------------------
    using Callback = std::function<void(const StatusCode, const Command&)>;

    //==============================================================================
    //! \brief Raw function pointer type of the reply callback function objects
    //! \note It is necessary to declare this to compare callback function objects
    //! with each other.
    //------------------------------------------------------------------------------
    using RawCallback = void (*)(const StatusCode, const Command&);

public:
    //========================================
    //! \brief Default wait time for execution progress status.
    //--------------------------------------
    static MICROVISION_SDK_API constexpr uint32_t defaultWaitTimeInMs = 500U;

public:
    //==============================================================================
    //! \brief Constructor
    //------------------------------------------------------------------------------
    Command(std::unique_ptr<CommandInterfaceTag>&& originatorTag,
            ConfigurationUPtr&& request,
            ConfigurationUPtr&& response);

    //==============================================================================
    //! \brief Copy constructor
    //------------------------------------------------------------------------------
    Command(const Command&) = delete;

    //==============================================================================
    //! \brief Move constructor
    //------------------------------------------------------------------------------
    Command(Command&&) = delete;

    //==============================================================================
    //! \brief Destructor
    //------------------------------------------------------------------------------
    virtual ~Command();

public: // Getter
    //========================================
    //! \brief Get the current request type of command.
    //! \returns The current request type;
    //--------------------------------------
    std::string getRequestType() const;

    //========================================
    //! \brief Get number of executions of this command.
    //! \returns Number of executions;
    //--------------------------------------
    uint32_t getNbOfExecutions() const;

    //==============================================================================
    //! \brief Get the value of specific property of the request instance using its config ID.
    //! \param[in]  id     ID of the property
    //! \param[out] value  Value of the property
    //! \returns Whether retrieving the property value was successful.
    //------------------------------------------------------------------------------
    template<typename ValueType>
    bool getRequestProperty(const std::string& id, ValueType& value) const
    {
        if (this->m_request)
        {
            return this->m_request->tryGetValueOrDefault(id, value);
        }

        return false;
    }

    //==============================================================================
    //! \brief Get the value of a specific property of the response instance using its config ID.
    //! \param[in]  id      ID of the property
    //! \param[out]  value  Value of the property
    //! \returns Whether returning the property value was successful.
    //------------------------------------------------------------------------------
    template<typename ValueType>
    bool getResponseProperty(const std::string& id, ValueType& value) const
    {
        if (this->m_response)
        {
            return this->m_response->tryGetValueOrDefault(id, value);
        }

        return false;
    }

    //========================================
    //! \brief Get the progress status of command execution.
    //! If the \c waitTimeInMs is greater than zero, than the calling this
    //! function blocks until, either the \c waitTimeInMs milliseconds have passed since the call
    //! or if a status code is available.
    //!
    //! \param[in] waitTimeInMs  The maximum duration which it is waited on a progress status.
    //! \returns Progress status of command execution.
    //--------------------------------------
    virtual StatusCode getExecutionProgressStatus(const uint32_t waitTimeInMs = defaultWaitTimeInMs) const = 0;

    //========================================
    //! \brief Get the error status of command execution.
    //! \returns Error status of command execution.
    //--------------------------------------
    virtual StatusCode getExecutionErrorStatus() const = 0;

public: // Setter
    //==============================================================================
    //! \brief Set a specific property of the response property.
    //! \param[in]  id     ID of of the property
    //! \param[in]  value  Value of the property
    //! \returns Whether setting the response property has been successful.
    //------------------------------------------------------------------------------
    template<typename ValueType>
    bool setRequestProperty(const std::string& id, const ValueType& value)
    {
        if (this->m_request)
        {
            return m_request->trySetValue(id, value);
        }
        return false;
    }

public: // Command execution
    //========================================
    //! \brief Execute this command.
    //--------------------------------------
    void execute();

    //========================================
    //! \brief Abort execution of this command.
    //--------------------------------------
    void abort();

public: // Callback managing
    //==============================================================================
    //! \brief Add a callback to this command.
    //! It can be invoked during initialization, execution, termination or clean up
    //! of the command.
    //! \param[in]  callback  Callback to add.
    //------------------------------------------------------------------------------
    void addCallback(const Callback& callback);

    //==============================================================================
    //! \brief Remove a callback from this command.
    //! \param[in]  callback  Callback to remove.
    //------------------------------------------------------------------------------
    void removeCallback(const Callback& callback);

protected:
    //========================================
    //! \brief Implementation of command execution.
    //--------------------------------------
    virtual void executeInternal() = 0;

    //========================================
    //! \brief Implementation of command execution abortion.
    //--------------------------------------
    virtual void abortInternal() = 0;

private:
    //==============================================================================
    //! \brief Search for a specific callback in the list of callbacks.
    //! \param[in]  callback  Callback to be searched.
    //! \returns The iterator to the matched callback. If it is not found it returns the end-iterator.
    //------------------------------------------------------------------------------
    std::vector<Callback>::iterator findCallback(const Callback& callback);

private:
    //==============================================================================
    //! \brief Logger
    //------------------------------------------------------------------------------
    static MICROVISION_SDK_API logging::LoggerSPtr logger;

protected:
    //========================================
    //! \brief Tag to the originator of this command.
    //--------------------------------------
    std::unique_ptr<CommandInterfaceTag> m_originatorTag;

    //==============================================================================
    //! \brief Request instance.
    //------------------------------------------------------------------------------
    ConfigurationPtr m_request;

    //==============================================================================
    //! \brief Response instance.
    //------------------------------------------------------------------------------
    ConfigurationPtr m_response;

    //========================================
    //! \brief Number of executions.
    //--------------------------------------
    uint32_t m_nbOfExecutions;

    //==============================================================================
    //! \brief Vector with callbacks.
    //------------------------------------------------------------------------------
    std::vector<Callback> m_callbacks;
};

//========================================
//! \brief Shared pointer to command.
//--------------------------------------
using CommandPtr = std::shared_ptr<Command>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
