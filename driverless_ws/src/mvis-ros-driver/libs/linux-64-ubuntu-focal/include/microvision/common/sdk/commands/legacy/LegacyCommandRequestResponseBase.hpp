//==============================================================================
//! \file
//!
//! \brief Legacy ECU command ID used for requests/responses to/from to an Appbase ECU.
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

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/config/Configuration.hpp>
#include <microvision/common/sdk/config/EnumConfigurationPropertyOf.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Represents the legacy command base, used for ECU, LUX and Mini LUX commands.
//! All legacy Appbase ECU requests and replies are sharing a common trait: They all contain a
//! command ID which is used to signal the command type to the Appbase ECU which receives the
//! request or sends the reply, respectively. As all reponse/reply classes it inherits
//! from \c Configuration because it is the data structure best suited for this case.
//------------------------------------------------------------------------------
class LegacyCommandRequestResponseBase : public Configuration
{
public:
    //========================================
    //! \brief Legacy command ID
    //--------------------------------------
    enum class CommandId : uint16_t
    {
        // ECU commands
        CmdManagerSetFilter     = 0x0005, ///< Sets the data type filter, EMPTY REPLY
        CmdManagerAppBaseCtrl   = 0x000B, //!< EMPTY REPLY
        CmdManagerAppBaseStatus = 0x000C, //!<

        // LUX3 commands
        CmdLuxReset                  = 0x0000, //!< ID of the Reset command, NO REPLY!
        CmdLuxGetStatus              = 0x0001, //!< ID of the GetStatus command
        CmdLuxSaveConfig             = 0x0004, //!< ID of the SaveConfig command, EMPTY REPLY!
        CmdLuxSetParameter           = 0x0010, //!< sets a parameter in the sensor, EMPTY REPLY!
        CmdLuxGetParameter           = 0x0011, //!< reads a parameter from the sensor
        CmdLuxResetDefaultParameters = 0x001A, //!< resets all parameters to the factory defaults, EMPTY REPLY!
        CmdLuxStartMeasure           = 0x0020, //!< starts the measurement with the currently configured settings
        CmdLuxStopMeasure            = 0x0021, //!< stops the measurement
        CmdLuxSetNtpTimestampSync    = 0x0034 //!< set the complete NtpTime stamp, EMPTY REPLY!
    };

public:
    //========================================
    //! \brief Property ID for the legacy command ID.
    //--------------------------------------
    static MICROVISION_SDK_API constexpr const char* commandIdPropId{"legacy_command_id"};

public:
    //========================================
    //! \brief Constructor
    //! \param[in]  id   Command ID to use.
    //--------------------------------------
    LegacyCommandRequestResponseBase(CommandId id = CommandId::CmdLuxReset);

    //========================================
    //! \brief Copy constructor
    //! \param[in]   other  The instance which shall be copied.
    //--------------------------------------
    LegacyCommandRequestResponseBase(const LegacyCommandRequestResponseBase& other);

    //========================================
    //! \brief Move constructor (deleted)
    //--------------------------------------
    LegacyCommandRequestResponseBase(LegacyCommandRequestResponseBase&&) = delete;

    //========================================
    //! \brief Destructor.
    //--------------------------------------
    ~LegacyCommandRequestResponseBase() override;

public:
    //========================================
    //! \brief Get command ID.
    //! \returns The command ID.
    //--------------------------------------
    EnumConfigurationPropertyOfType<CommandId>& getCommandId();

protected:
    //========================================
    //! \brief Legacy command ID.
    //--------------------------------------
    EnumConfigurationPropertyOfType<CommandId> m_commandId;
};

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
