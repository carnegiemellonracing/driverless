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
//! \date Sept 03, 2018
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/DataTypeId.hpp>
#include <microvision/common/sdk/datablocks/PositionWgs84.hpp>
#include <microvision/common/sdk/datablocks/DataContainerBase.hpp>

#include <microvision/common/sdk/misc/SdkHash.hpp>

#include <array>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Contains a sequence of wgs84 positions
//------------------------------------------------------------------------------
class PositionWgs84Sequence3510 final : public SpecializedDataContainer
{
    template<class ContainerType, DataTypeId::DataType id>
    friend class Importer;
    template<class ContainerType, DataTypeId::DataType id>
    friend class Exporter;

public:
    constexpr static const char* const containerType{"sdk.specialcontainer.positionwgs84sequence3510"};
    static constexpr uint8_t nbOfReserved = 4;

public: // type declaration
    using ReservedArray = std::array<uint32_t, nbOfReserved>;

    static constexpr uint64_t getClassIdHashStatic() { return hash(containerType); }

public:
    enum class SourceType : uint8_t
    {
        Mission = 0,
        Unknown = 255
    };

public:
    PositionWgs84Sequence3510();
    ~PositionWgs84Sequence3510() override = default;

public:
    uint64_t getClassIdHash() const override { return getClassIdHashStatic(); }

public:
    void setPositionSequence(const std::vector<PositionWgs84>& positionSequence)
    {
        this->m_positionSequence = positionSequence;
    }
    void setTimestamp(const NtpTime& timestamp) { this->m_timestamp = timestamp; }
    void setSourceType(const SourceType sourceType) { this->m_sourceType = sourceType; }
    void setId(const uint32_t id) { this->m_id = id; }
    const std::vector<PositionWgs84>& getPositionSequence() const { return m_positionSequence; }
    const NtpTime& getTimestamp() const { return m_timestamp; }
    SourceType getSourceType() const { return m_sourceType; }
    uint32_t getId() const { return m_id; }

    //==============================================================================
    //!\brief Returns value from m_reserved array at specified position \a idx.
    //!\param[in] idx  Index specifying position in m_reserved array.
    //!\return Value of m_reserved array at position \a idx. Returns last element of array, if \a idx >= \a nbOfReserved.
    //------------------------------------------------------------------------------
    uint32_t getReserved(const uint_fast8_t idx) const;

protected:
    std::vector<PositionWgs84> m_positionSequence{0}; //!< Sequence of wgs84 positions
    NtpTime m_timestamp{}; //!< Timestamp of of this data type
    SourceType m_sourceType{SourceType::Unknown}; //!< Indicates source of this data type
    uint32_t m_id{0}; //!< Identifier of position sequence

private:
    ReservedArray m_reserved{{}};

}; //PositionWgs84Sequence3510

//==============================================================================

bool operator==(const PositionWgs84Sequence3510& lhs, const PositionWgs84Sequence3510& rhs);
bool operator!=(const PositionWgs84Sequence3510& lhs, const PositionWgs84Sequence3510& rhs);

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
