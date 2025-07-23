//==============================================================================
//! \file
//!
//! \brief Appbase ECU set filter request.
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

#include <microvision/common/sdk/DataTypeId.hpp>
#include <microvision/common/sdk/commands/legacy/LegacyCommandRequestResponseBase.hpp>
#include <microvision/common/sdk/config/ConfigurationPropertyOfType.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Represents a filter request for the Appbase ECU
//!
//! It notifies the Appbase ECU to filter out specific datatypes from its output.
//! These datatypes are represented via ID ranges.
//------------------------------------------------------------------------------
class EcuSetFilterRequest : public LegacyCommandRequestResponseBase
{
public:
    //========================================
    //! \brief Base type
    //--------------------------------------
    using BaseType = LegacyCommandRequestResponseBase;

    //========================================
    //! \brief Key type.
    //--------------------------------------
    using KeyType = LegacyCommandRequestResponseBase::CommandId;

    //========================================
    //! \brief Data type ID range.
    //--------------------------------------
    using Range = std::pair<DataTypeId, DataTypeId>;

    //========================================
    //! \brief Vector of data type ID ranges.
    //--------------------------------------
    using RangesVector = std::vector<Range>;

public:
    //========================================
    //! \brief The underlying legacy ECU command ID
    //--------------------------------------
    static constexpr KeyType key = KeyType::CmdManagerSetFilter;

    //========================================
    //! \brief Property ID of ranges vector.
    //--------------------------------------
    static MICROVISION_SDK_API constexpr const char* rangesVectorPropId{"ranges_vector"};

    //========================================
    //! \brief Request type.
    //--------------------------------------
    static MICROVISION_SDK_API const std::string type;

public:
    //========================================
    //! \brief Constructor.
    //! \param[in] rangesVector  Vector of data type ID ranges.
    //--------------------------------------
    EcuSetFilterRequest(const RangesVector& rangesVector
                        = {{DataTypeId::DataType_Unknown, DataTypeId::DataType_LastId}});

    //========================================
    //! \brief Copy constructor.
    //! \param[in]  other  Other instance
    //--------------------------------------
    EcuSetFilterRequest(const EcuSetFilterRequest& other);

    //========================================
    //! \brief Move constructor (deleted)
    //--------------------------------------
    EcuSetFilterRequest(EcuSetFilterRequest&&) = delete;

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
    //! \brief Get ranges vector.
    //! \returns Ranges vector.
    //--------------------------------------
    ConfigurationPropertyOfType<RangesVector>& getRangesVector();

private:
    //========================================
    //! \brief Vector of data type ID ranges which are supposed to be filtered out.
    //--------------------------------------
    ConfigurationPropertyOfType<RangesVector> m_rangesVector;
};

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
