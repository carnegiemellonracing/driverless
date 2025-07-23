//==============================================================================
//! \file
//!
//! \brief Appbase ECU control request.
//!
//! $$MICROVISION_LICENSE_BEGIN$$
//! Copyright (c) 2025 MicroVision, Inc., Redmond, U.S.A.
//! All Rights Reserved.
//!
//! For more details, please refer to the accompanying file
//! License.txt.
//! $$MICROVISION_LICENSE_END$$
//!
//! \date Oct 24, 2021
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/commands/legacy/LegacyCommandRequestResponseBase.hpp>
#include <microvision/common/sdk/config/ConfigurationPropertyOfType.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Represents the Appbase ECU control request
//------------------------------------------------------------------------------
class AppbaseControlRequest final : public LegacyCommandRequestResponseBase
{
public:
    //========================================
    //! \brief ID of the control request.
    //--------------------------------------
    enum class AppbaseControlId : uint16_t
    {
        Invalid        = 0x0000,
        StartRecording = 0x0001,
        StopRecording  = 0x0002
    };

public:
    //========================================
    //! \brief Base type
    //--------------------------------------
    using BaseType = LegacyCommandRequestResponseBase;

    //========================================
    //! \brief Key type.
    //--------------------------------------
    using KeyType = LegacyCommandRequestResponseBase::CommandId;

public:
    //========================================
    //! \brief The underlying legacy ECU command ID
    //--------------------------------------
    static constexpr KeyType key{KeyType::CmdManagerAppBaseCtrl};

    //========================================
    //! \brief Property ID for the Appbase Control ID.
    //--------------------------------------
    static MICROVISION_SDK_API constexpr const char* appbaseControlIdPropId{"appbase_control_id"};

    //========================================
    //! \brief Property ID of the control data.
    //--------------------------------------
    static MICROVISION_SDK_API constexpr const char* dataPropId{"data"};

    //========================================
    //! \brief Request type.
    //--------------------------------------
    static MICROVISION_SDK_API const std::string type;

public:
    //========================================
    //! \brief Constructor
    //! \param[in]   appbaseControlId  Appbase control ID
    //! \param[in]   data              Control data
    //--------------------------------------
    AppbaseControlRequest(const AppbaseControlId& appbaseControlId = AppbaseControlId::Invalid,
                          const std::string& data                  = "");

    //========================================
    //! \brief Copy constructor
    //! \param[in]  other  Other instance
    //--------------------------------------
    AppbaseControlRequest(const AppbaseControlRequest& other);

    //========================================
    //! \brief Move constructor (deleted).
    //--------------------------------------
    AppbaseControlRequest(AppbaseControlRequest&&) = delete;

    //========================================
    //! \brief Destructor
    //--------------------------------------
    ~AppbaseControlRequest() override;

public:
    //========================================
    //! \brief Get type of this request.
    //! \returns Type
    //--------------------------------------
    const std::string& getType() const override;

    //========================================
    //! \brief Return copy of this request.
    //! \returns Fresh copy.
    //--------------------------------------
    ConfigurationPtr copy() const override;

    //========================================
    //! \brief Return serialized size.
    //! \returns Serialized size.
    //--------------------------------------
    std::size_t getSerializedSize() const override;

    //========================================
    //! \brief Serialize this request into stream.
    //! \param[out]  os  Output stream.
    //! \returns Whether serialization was successful.
    //--------------------------------------
    bool serialize(std::ostream& os) const override;

    //========================================
    //! \brief Deserialize this request from stream.
    //! \param[out]  is  Input stream.
    //! \returns Whether deserialization was successful.
    //--------------------------------------
    bool deserialize(std::istream& is) override;

public:
    //========================================
    //! \brief Get Appbase Control ID.
    //! \returns Appbase Control ID.
    //--------------------------------------
    EnumConfigurationPropertyOfType<AppbaseControlId>& getAppbaseControlId();

    //========================================
    //! \brief Get control data.
    //! \returns Control data.
    //--------------------------------------
    ConfigurationPropertyOfType<std::string>& getData();

private:
    //========================================
    //! \brief Appbase Control ID
    //--------------------------------------
    EnumConfigurationPropertyOfType<AppbaseControlId> m_appbaseControlId;

    //========================================
    //! \brief Appbase Control data
    //--------------------------------------
    ConfigurationPropertyOfType<std::string> m_data;
};

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
