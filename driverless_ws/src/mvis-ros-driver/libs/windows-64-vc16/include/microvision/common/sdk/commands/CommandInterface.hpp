//==============================================================================
//! \file
//!
//! \brief Interface for all devices which support commands.
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

#include <microvision/common/sdk/commands/Command.hpp>
#include <microvision/common/sdk/commands/CommandInterfaceTag.hpp>
#include <microvision/common/logging/logging.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

class CommandInterface;

//==============================================================================
//! \brief  Interface which devices can implement to send command requests and receive responses.
//------------------------------------------------------------------------------
class CommandInterface
{
public:
    //========================================
    //! \brief Constructor
    //--------------------------------------
    CommandInterface();

    //========================================
    //! \brief Copy constructor (deleted)
    //--------------------------------------
    CommandInterface(const CommandInterface&) = delete;

    //========================================
    //! \brief Move constructor (deleted)
    //--------------------------------------
    CommandInterface(CommandInterface&&) = delete;

    //==============================================================================
    //! \brief Destructor
    //------------------------------------------------------------------------------
    virtual ~CommandInterface();

public:
    //==============================================================================
    //! \brief Return all available request types.
    //! \returns The vector containing the available request types.
    //------------------------------------------------------------------------------
    virtual const std::vector<std::string> getAvailableRequestTypes() const = 0;

    //==============================================================================
    //! \brief Create command.
    //! \param[in]  requestType  Request type which shall be used for this command.
    //! \returns The pointer to the created command.
    //------------------------------------------------------------------------------
    CommandPtr createCommand(const std::string& requestType);

    //========================================
    //! \brief Create command and set an arbitrary number of its properties.
    //! \tparam      StringType  Type which is implicitely convertible to std::string.
    //! \tparam      ValueType   Type of the property value.
    //! \tparam      RestTypes   Types of remaining ID/value pairs.
    //! \param[in]  requestType  Desired request type of created command.
    //! \param[in]  propertyId   ID of property which shall be set.
    //! \param[in]  propertyValue  Value to which property shall be set.
    //! \param[out] restValues   Remaining ID/value pairs.
    //--------------------------------------
    template<typename StringType, typename ValueType, typename... RestTypes>
    CommandPtr createCommand(const std::string requestType,
                             const StringType& propertyId,
                             const ValueType& propertyValue,
                             RestTypes... restValues)
    {
        auto command = createCommand(requestType, restValues...);

        if (!command)
        {
            return nullptr;
        }
        else if (!command->setRequestProperty(propertyId, propertyValue))
        {
            LOGWARNING(logger, "Setting property " << propertyId << " failed!");
        }

        return command;
    }

    //========================================
    //! \brief Get CommandInterfaceTag instance which monitors whether this instance has been destructed.
    //! \returns CommandInterfaceTag instance.
    //--------------------------------------
    CommandInterfaceTagUPtr getTag();

private:
    //========================================
    //! \brief Logger
    //--------------------------------------
    static MICROVISION_SDK_API logging::LoggerSPtr logger;

private:
    //========================================
    //! \brief Internal implementation of factory method to create commands.
    //! \param[in]  requestType  Request type which shall be used for this command.
    //! \returns Pointer to created command.
    //--------------------------------------
    virtual CommandPtr createCommandInternal(const std::string& requestType) = 0;

    //========================================
    //! \brief Shared variable which represents the expiration status of this command interface.
    //--------------------------------------
    std::shared_ptr<bool> m_isAlive;
};

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
