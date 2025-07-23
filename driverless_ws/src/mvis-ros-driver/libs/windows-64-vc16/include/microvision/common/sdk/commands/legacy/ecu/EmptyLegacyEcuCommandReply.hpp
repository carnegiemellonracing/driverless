//==============================================================================
//! \file
//!
//! \brief Empty reply coming from Appbase ECUs
//!
//! $$MICROVISION_LICENSE_BEGIN$$
//! Copyright (c) 2025 MicroVision, Inc., Redmond, U.S.A.
//! All Rights Reserved.
//!
//! For more details, please refer to the accompanying file
//! License.txt.
//! $$MICROVISION_LICENSE_END$$
//!
//! \date Nov 2, 2021
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
//! \brief Represents an empty reply from the Appbase ECU.
//------------------------------------------------------------------------------
class EmptyLegacyEcuCommandReply : public LegacyCommandRequestResponseBase
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

public:
    //========================================
    //! \brief Reply type.
    //--------------------------------------
    static MICROVISION_SDK_API const std::string type;

public:
    //========================================
    //! \brief Constructor
    //! \param[in]  commandId  Command ID of the request this reply belongs to.
    //--------------------------------------
    EmptyLegacyEcuCommandReply(const KeyType& commandId = KeyType::CmdManagerSetFilter);

    //========================================
    //! \brief Copy Constructor
    //! \param[in]  other  Other instance.
    //--------------------------------------
    EmptyLegacyEcuCommandReply(const EmptyLegacyEcuCommandReply& other);

    //========================================
    //! \brief Move constructor (deleted).
    //--------------------------------------
    EmptyLegacyEcuCommandReply(EmptyLegacyEcuCommandReply&&) = delete;

    //========================================
    //! \brief Destructor.
    //--------------------------------------
    ~EmptyLegacyEcuCommandReply() override;

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
};

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
