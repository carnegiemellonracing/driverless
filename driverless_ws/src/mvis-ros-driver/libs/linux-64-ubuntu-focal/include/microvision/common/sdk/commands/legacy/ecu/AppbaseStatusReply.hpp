//==============================================================================
//! \file
//!
//! \brief Appbase ECU status reply.
//!
//! $$MICROVISION_LICENSE_BEGIN$$
//! Copyright (c) 2025 MicroVision, Inc., Redmond, U.S.A.
//! All Rights Reserved.
//!
//! For more details, please refer to the accompanying file
//! License.txt.
//! $$MICROVISION_LICENSE_END$$
//!
//! \date Nov 1, 2021
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/commands/legacy/LegacyCommandRequestResponseBase.hpp>
#include <microvision/common/sdk/commands/legacy/ecu/AppbaseStatusRequest.hpp>
#include <microvision/common/sdk/config/ConfigurationPropertyOfType.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Represents the reply of the Appbase ECU upon a status request.
//------------------------------------------------------------------------------
class AppbaseStatusReply : public LegacyCommandRequestResponseBase
{
public:
    //========================================
    //! \brief Base type
    //--------------------------------------
    using BaseType = LegacyCommandRequestResponseBase;

    //========================================
    //! \brief Key type.
    //--------------------------------------
    using KeyType = BaseType::CommandId;

    //========================================
    //! \brief Status ID.
    //--------------------------------------
    using AppbaseStatusId = AppbaseStatusRequest::AppbaseStatusId;

public:
    //========================================
    //! \brief The underlying legacy ECU command ID
    //--------------------------------------
    static constexpr KeyType key{AppbaseStatusRequest::key};

    //========================================
    //! \brief Reply Type.
    //--------------------------------------
    static MICROVISION_SDK_API constexpr const char* id{"appbase_status_reply"};

    //========================================
    //! \brief Property ID of the status ID.
    //--------------------------------------
    static MICROVISION_SDK_API constexpr const char* appbaseStatusIdPropId{"appbase_status_id"};

    //========================================
    //! \brief Property ID of data.
    //--------------------------------------
    static MICROVISION_SDK_API constexpr const char* dataPropId{"data"};

    //========================================
    //! \brief Request type.
    //! \note This static class variable only exist to comply with the Configuration interface.
    //! \todo Mark this as deprecated.
    //--------------------------------------
    static MICROVISION_SDK_API const std::string type;

    //========================================
    //! \brief Maximum size of the data string.
    //--------------------------------------
    static constexpr uint16_t maxDataStringSize{10000};

public:
    //========================================
    //! \brief Constructor
    //! \param[in]  statusId  Status ID
    //! \param[in]  data      Data
    //--------------------------------------
    AppbaseStatusReply(const AppbaseStatusId& statusId = AppbaseStatusId::Recording, const std::string& data = "");

    //========================================
    //! \brief Copy constructor
    //! \param[in]  other  Other instance.
    //--------------------------------------
    AppbaseStatusReply(const AppbaseStatusReply& other);

    //========================================
    //! \brief Move constructor (deleted)
    //--------------------------------------
    AppbaseStatusReply(AppbaseStatusReply&&) = delete;

    //========================================
    //! \brief Destructor
    //--------------------------------------
    ~AppbaseStatusReply() override;

public:
    //========================================
    //! \brief Get type of this reply.
    //! \returns Type
    //--------------------------------------
    const std::string& getType() const override;

    //========================================
    //! \brief Return copy of this reply.
    //! \returns Fresh copy.
    //--------------------------------------
    ConfigurationPtr copy() const override;

    //========================================
    //! \brief Return serialized size.
    //! \returns Serialized size.
    //--------------------------------------
    std::size_t getSerializedSize() const override;

    //========================================
    //! \brief Serialize this reply into stream.
    //! \param[out]  os  Output stream.
    //! \returns Whether serialization was successful.
    //--------------------------------------
    bool serialize(std::ostream& os) const override;

    //========================================
    //! \brief Deserialize this reply from stream.
    //! \param[out]  is  Input stream.
    //! \returns Whether deserialization was successful.
    //--------------------------------------
    bool deserialize(std::istream& is) override;

public:
    //========================================
    //! \brief Get Appbase status id.
    //! \returns Status ID.
    //--------------------------------------
    EnumConfigurationPropertyOfType<AppbaseStatusId>& getAppBaseStatusId();

    //========================================
    //! \brief Get Appbase status data.
    //! \returns Status data.
    //--------------------------------------
    ConfigurationPropertyOfType<std::string>& getData();

private:
    //========================================
    //! \brief Appbase status ID.
    //--------------------------------------
    EnumConfigurationPropertyOfType<AppbaseStatusId> m_appbaseStatusId;

    //========================================
    //! \brief Appbase status data.
    //--------------------------------------
    ConfigurationPropertyOfType<std::string> m_data;
};

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
