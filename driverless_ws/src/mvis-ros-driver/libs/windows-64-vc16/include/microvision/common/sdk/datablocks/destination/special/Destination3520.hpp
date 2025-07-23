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
//! \date Sept 05, 2018
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/DataContainerBase.hpp>
#include <microvision/common/sdk/datablocks/PositionWgs84.hpp>

#include <microvision/common/sdk/misc/SdkHash.hpp>

#include <array>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief target position as GPS coordinate
//!
//! this data type was generated with the interface to BestMile for fleet management
//! in mind [https://developer.bestmile.com/v1/docs/performing-a-mission#section-mission-fields-overview],
//! it's similar to Destination field
//------------------------------------------------------------------------------
class Destination3520 final : public SpecializedDataContainer
{
    template<class ContainerType, DataTypeId::DataType id>
    friend class Importer;
    template<class ContainerType, DataTypeId::DataType id>
    friend class Exporter;

public:
    static constexpr uint8_t nbOfReserved = 4;

public: // type declaration
    using ReservedArray = std::array<uint32_t, nbOfReserved>;

public:
    //! Possible sources of destination data
    //! This type is serialized as UINT8, 255 is the maximum value
    enum class SourceType : uint8_t
    {
        Mission = 0,
        Unknown = 255
    };

    //! Possible type of destination
    //! This type is serialized as UINT8, 255 is the maximum value
    enum class DestinationType : uint8_t
    {
        Station          = 0,
        IntermediateHalt = 1,
        Unknown          = 255
    };

    //! Possible purpose types of destination data
    //! This type is serialized as UINT8, 255 is the maximum value
    enum class PurposeType : uint8_t
    {
        PassengerTransport = 0,
        Maintenance        = 1,
        Unknown            = 255
    };

public:
    Destination3520();
    ~Destination3520() override = default;

public:
    constexpr static const char* const containerType{"sdk.specialcontainer.destination3520"};

    static constexpr uint64_t getClassIdHashStatic() { return hash(containerType); }

    uint64_t getClassIdHash() const override { return getClassIdHashStatic(); }

public: // setters and getters
    void setTargetPosition(const PositionWgs84& targetPosition) { this->m_targetPosition = targetPosition; }
    void setTargetId(const int64_t targetId) { this->m_targetId = targetId; }
    void setTimestamp(const NtpTime& timestamp) { this->m_timestamp = timestamp; }
    void setSourceType(const SourceType sourceType) { this->m_sourceType = sourceType; }
    void setDestinationType(const DestinationType destinationType) { this->m_destinationType = destinationType; }
    void setPurposeType(const PurposeType purposeType) { this->m_purposeType = purposeType; }
    void setId(const uint32_t id) { this->m_id = id; }

public:
    const PositionWgs84& getTargetPosition() const { return m_targetPosition; }
    int64_t getTargetId() const { return m_targetId; }
    const NtpTime& getTimestamp() const { return m_timestamp; }
    SourceType getSourceType() const { return m_sourceType; }
    DestinationType getDestinationType() const { return m_destinationType; }
    PurposeType getPurposeType() const { return m_purposeType; }
    uint32_t getId() const { return m_id; }

    //==============================================================================
    //!\brief Returns value from m_reserved array at specified position \a idx.
    //!\param[in] idx  Index specifying position in \a m_reserved array.
    //!\return Value of \a m_reserved array at position \a idx.
    //!        Returns last element of array, if \a idx >= \a nbOfReserved.
    //------------------------------------------------------------------------------
    uint32_t getReserved(const uint_fast8_t idx) const;

protected:
    PositionWgs84 m_targetPosition{}; //!< Wgs84 position of target destination
    int64_t m_targetId{0}; //!< Defines id of the target e.g. parking space id, or edge id
    NtpTime m_timestamp{}; //!< Timestamp of of this data type
    SourceType m_sourceType{SourceType::Unknown}; //!< Indicates source of this data type
    DestinationType m_destinationType{
        DestinationType::Unknown}; //!< Indicates the type of this destination e.g. a station
    PurposeType m_purposeType{PurposeType::Unknown}; //!< Indicates the purpose of this destination e.g. maintenance
    uint32_t m_id{0}; //!< Identifier of destination, e.g. mission id

private:
    ReservedArray m_reserved{{}};
}; // Destination3520

//==============================================================================

bool operator==(const Destination3520& lhs, const Destination3520& rhs);
bool operator!=(const Destination3520& lhs, const Destination3520& rhs);

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
