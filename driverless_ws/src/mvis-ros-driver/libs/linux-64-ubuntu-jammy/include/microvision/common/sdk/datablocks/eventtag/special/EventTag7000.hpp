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
//! \date Mar 21, 2018
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

//! \brief Event Tag
class EventTag7000 final : public SpecializedDataContainer
{
    template<class ContainerType, DataTypeId::DataType id>
    friend class Importer;
    template<class ContainerType, DataTypeId::DataType id>
    friend class Exporter;

public:
    static const uint8_t nbOfReserved = 16;

public:
    using ReservedArray = std::array<char, nbOfReserved>;

    enum class TagClass : uint16_t
    {
        Unknown    = 0x0000U,
        Pedestrian = 0x0001U
    };

    enum class Flags : uint8_t
    {
        IsEnd   = 0x01U,
        IsPulse = 0x02U
    };

public:
    constexpr static const char* const containerType{"sdk.specialcontainer.eventtag7000"};

    static constexpr uint64_t getClassIdHashStatic() { return hash(containerType); }

public:
    EventTag7000()           = default;
    ~EventTag7000() override = default;

public:
    uint64_t getClassIdHash() const override { return getClassIdHashStatic(); }

public:
    NtpTime getTagStart() const { return m_tagStart; }
    NtpTime getTagEnd() const { return m_tagEnd; }
    uint8_t getFlags() const { return m_flags; }
    TagClass getTagClass() const { return m_tagClass; }
    uint32_t getTagId() const { return m_tagId; }
    const std::string& getTagString() const { return m_tagString; }
    std::string& getTagString() { return m_tagString; }
    const PositionWgs84& getWgs84() const { return m_wgs84; }
    PositionWgs84& getWgs84() { return m_wgs84; }
    uint64_t getRoiWidth() const { return m_roiWidth; }
    uint64_t getRoiLength() const { return m_roiLength; }

    bool isEndOfEvent() const { return ((m_flags & static_cast<uint8_t>(Flags::IsEnd)) != 0); }
    bool isPulseEvent() const { return ((m_flags & static_cast<uint8_t>(Flags::IsPulse)) != 0); }

    char getReserved(const uint8_t idx) const { return m_reserved.at(idx); }
    const ReservedArray& getReserved() const { return m_reserved; }

public:
    void setTagStart(const NtpTime newTagStart) { m_tagStart = newTagStart; }
    void setTagEnd(const NtpTime newTagEnd) { m_tagEnd = newTagEnd; }
    void setFlags(const uint8_t newFlags) { m_flags = newFlags; }
    void setTagClass(const TagClass newTagClass) { m_tagClass = newTagClass; }
    void setTagId(const uint32_t newTagId) { m_tagId = newTagId; }
    void setTagString(const std::string& newTagString) { m_tagString = newTagString; }
    void setWgs84(const PositionWgs84& newWgs84) { m_wgs84 = newWgs84; }
    void setRoiWidth(const uint64_t newRoiWidth) { m_roiWidth = newRoiWidth; }
    void setRoiLength(const uint64_t newRoiLength) { m_roiLength = newRoiLength; }

    void setIsEndOfEvent() { m_flags |= static_cast<uint8_t>(Flags::IsEnd); }
    void unsetIsEndOfEvent() { m_flags &= static_cast<uint8_t>(~static_cast<uint8_t>(Flags::IsEnd)); }

    void setIsPulseEvent() { m_flags |= static_cast<uint8_t>(Flags::IsPulse); }
    void unsetIsPulseEvent() { m_flags &= static_cast<uint8_t>(~static_cast<uint8_t>(Flags::IsPulse)); }

protected:
    NtpTime m_tagStart{}; //!< Timestamp for the start of the event.
    NtpTime m_tagEnd{}; //!< Timestamp for the start of the event.
    uint8_t m_flags{0U}; //!< Flags indicating the start or end of the event.
    TagClass m_tagClass{TagClass::Unknown}; //!< Defines the classes of the tags.
    uint32_t m_tagId{0U}; //!< The id for the tags.
    std::string m_tagString{}; //!< String to store arbitrary notes, comments or additional information.
    PositionWgs84 m_wgs84{}; //!< Position of the tag.
    uint64_t m_roiWidth{0U}; //!< ROI width for the event around the PositionGPS
    uint64_t m_roiLength{0U}; //!< ROI length for the event around the PositionGPS
    ReservedArray m_reserved{{0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U}};
}; // EventTag7000

//==============================================================================

bool operator==(const EventTag7000& lhs, const EventTag7000& rhs);
bool operator!=(const EventTag7000& lhs, const EventTag7000& rhs);

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
