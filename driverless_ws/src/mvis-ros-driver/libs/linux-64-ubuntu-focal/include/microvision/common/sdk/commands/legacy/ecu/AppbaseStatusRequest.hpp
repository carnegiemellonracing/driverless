//==============================================================================
//! \file
//!
//! \brief Appbase ECU status request.
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

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Represents an Appbase ECU status request.
//------------------------------------------------------------------------------
class AppbaseStatusRequest : public LegacyCommandRequestResponseBase
{
public:
    //========================================
    //! \brief Represents Appbase status ID.
    //--------------------------------------
    enum class AppbaseStatusId : uint8_t
    {
        Recording = 0x01
    };

public:
    //========================================
    //! \brief Base type
    //--------------------------------------
    using BaseType = LegacyCommandRequestResponseBase;

    //========================================
    //! \brief Key type.
    //--------------------------------------
    using KeyType = BaseType::CommandId;

public:
    //========================================
    //! \brief The underlying legacy ECU command ID
    //--------------------------------------
    static constexpr KeyType key{KeyType::CmdManagerAppBaseStatus};

    //========================================
    //! \brief Property ID of Appbase status ID.
    //--------------------------------------
    static MICROVISION_SDK_API constexpr const char* appbaseStatusIdPropId{"appbase_status_id"};

    //========================================
    //! \brief Request type.
    //--------------------------------------
    static MICROVISION_SDK_API const std::string type;

public:
    //========================================
    //! \brief Constructor
    //! \param[in] statusId  Status ID
    //--------------------------------------
    AppbaseStatusRequest(const AppbaseStatusId statusId = AppbaseStatusId::Recording);

    //========================================
    //! \brief Copy constructor
    //! \param[in]  other  Other instance.
    //--------------------------------------
    AppbaseStatusRequest(const AppbaseStatusRequest& other);

    //========================================
    //! \brief Move constructor (deleted).
    //--------------------------------------
    AppbaseStatusRequest(AppbaseStatusRequest&&) = delete;

    //========================================
    //! \brief Destructor.
    //--------------------------------------
    ~AppbaseStatusRequest() override;

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
    //! \brief Get Appbase status ID.
    //! \returns Appbase status ID.
    //--------------------------------------
    EnumConfigurationPropertyOfType<AppbaseStatusId>& getAppbaseStatusId();

private:
    //========================================
    //! \brief Appbase status ID.
    //--------------------------------------
    EnumConfigurationPropertyOfType<AppbaseStatusId> m_appbaseStatusId;
};

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
