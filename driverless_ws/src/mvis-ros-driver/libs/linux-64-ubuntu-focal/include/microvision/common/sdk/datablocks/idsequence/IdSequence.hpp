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
//! \date 07.November 2018
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>
#include <microvision/common/sdk/datablocks/DataContainerBase.hpp>

#include <microvision/common/sdk/misc/SdkHash.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Sequence of mission ids
//!
//! Special data type: \ref microvision::common::sdk::IdSequence3500
//------------------------------------------------------------------------------
class IdSequence final : public DataContainerBase
{
    template<class ContainerType, DataTypeId::DataType id>
    friend class Importer;
    template<class ContainerType, DataTypeId::DataType id>
    friend class Exporter;

public:
    static constexpr uint8_t nbOfReserved = 4;

public:
    using ReservedArray    = std::array<uint32_t, nbOfReserved>;
    using IdSequenceVector = std::vector<int64_t>;

public:
    constexpr static const char* const containerType{"sdk.generalcontainer.idsequence"};

    static constexpr uint64_t getClassIdHashStatic() { return hash(containerType); }

    //! Possible sources for id sequence data
    //! This type is serialized as UINT8, 255 is the maximum value
    enum class SourceType : uint8_t
    {
        Mission = 0,
        Unknown = 255
    };

public:
    IdSequence()           = default;
    ~IdSequence() override = default;

    IdSequence& operator=(const IdSequence& other) = default;

public:
    uint64_t getClassIdHash() const override { return getClassIdHashStatic(); }

public: // setters and getters
    void setIdSequence(const IdSequenceVector& idSequence) { this->m_idSequence = idSequence; }
    void setTimestamp(const NtpTime& timestamp) { this->m_timestamp = timestamp; }
    void setSourceType(const SourceType sourceType) { this->m_sourceType = sourceType; }
    void setId(const uint32_t id) { this->m_id = id; }

    const IdSequenceVector& getIdSequence() const { return m_idSequence; }
    const NtpTime& getTimestamp() const { return m_timestamp; }
    SourceType getSourceType() const { return m_sourceType; }
    uint32_t getId() const { return m_id; }
    ReservedArray getReserved() const { return m_reserved; }

protected:
    IdSequenceVector m_idSequence{{}}; //!< Sequence of IDs
    NtpTime m_timestamp{}; //!< Timestamp of IdSequence
    SourceType m_sourceType{SourceType::Unknown}; //!< Indicates source of this data type
    uint32_t m_id{0}; //!< Identifier of ID sequence

private:
    ReservedArray m_reserved{{}};

}; // IdSequence

//==============================================================================

bool operator==(const IdSequence& is1, const IdSequence& is2);
bool operator!=(const IdSequence& is1, const IdSequence& is2);

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
