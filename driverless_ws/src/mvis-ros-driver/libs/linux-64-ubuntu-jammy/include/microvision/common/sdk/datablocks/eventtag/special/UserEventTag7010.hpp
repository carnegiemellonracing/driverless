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
//! \date Mar 15, 2019
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/eventtag/special/UserEventTag7010Types.hpp>

#include <microvision/common/sdk/datablocks/DataContainerBase.hpp>
#include <microvision/common/sdk/datablocks/PositionWgs84.hpp>

#include <microvision/common/sdk/misc/SdkHash.hpp>

#include <bitset>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================
//! \brief User Event Tag
//!
//! This class is used to capture manually entered events through the MVIS OS
//! Online Tagging website.
//! These events can be used to assist post processing in EVS
//------------------------------------------------------------------------------
class UserEventTag7010 final : public SpecializedDataContainer
{
    template<class ContainerType, DataTypeId::DataType id>
    friend class Importer;
    template<class ContainerType, DataTypeId::DataType id>
    friend class Exporter;

public:
    static const uint8_t nbOfReserved = 16;

private:
    static const uint8_t inqueryCategory    = 0x01U;
    static const uint8_t inqueryOccurence   = 0x02U;
    static const uint8_t inquerySeverity    = 0x04U;
    static const uint8_t inqueryTagValue    = 0x08U;
    static const uint8_t inqueryUserDefined = 0x10U;

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
    constexpr static const char* const containerType{"sdk.specialcontainer.usereventtag7010"};
    static constexpr uint64_t getClassIdHashStatic() { return hash(containerType); }

public: // C'tor / D'tor
    UserEventTag7010()           = default;
    ~UserEventTag7010() override = default;

public:
    uint64_t getClassIdHash() const override { return getClassIdHashStatic(); }

public: // Setter
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

    void setTagCategory(const usereventtag7010types::EventCategory category)
    {
        m_category = static_cast<uint8_t>(category);
        m_inquiry |= inqueryCategory;
    }
    void setTagCategoryRaw(const uint8_t category)
    {
        m_category = category;
        m_inquiry |= inqueryCategory;
    }
    void setTagOccurrence(const usereventtag7010types::TagOccurrence occurrence)
    {
        m_occurrence = occurrence;
        m_inquiry |= inqueryOccurence;
    }
    void setTagSeverity(const usereventtag7010types::TagSeverity severity)
    {
        m_severity = severity;
        m_inquiry |= inquerySeverity;
    }
    void setTagValue(const uint8_t byte)
    {
        m_value = byte;
        m_inquiry |= inqueryTagValue;
    }
    void setTagUserDefined() { m_inquiry |= inqueryUserDefined; }
    void setInquiry(const uint8_t inquiry) { m_inquiry = inquiry; }

public: // Re-setter
    void unsetTagCategory() { m_inquiry &= static_cast<uint8_t>(~inqueryCategory); }
    void unsetTagOccurrence() { m_inquiry &= static_cast<uint8_t>(~inqueryOccurence); }
    void unsetTagSeverity() { m_inquiry &= static_cast<uint8_t>(~inquerySeverity); }
    void unsetTagValue() { m_inquiry &= static_cast<uint8_t>(~inqueryTagValue); }
    void unsetTagUserDefined() { m_inquiry &= static_cast<uint8_t>(~inqueryUserDefined); }

public: // Getter
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

    usereventtag7010types::EventCategory getTagCategory() const
    {
        if (!hasTagUserDefined())
        {
            return static_cast<usereventtag7010types::EventCategory>(m_category);
        }
        else
        {
            return usereventtag7010types::EventCategory::UserDefined;
        }
    }

    uint8_t getTagRawCategory() const { return m_category; }
    usereventtag7010types::TagOccurrence getTagOccurrence() const { return m_occurrence; }
    usereventtag7010types::TagSeverity getTagSeverity() const { return m_severity; }
    uint8_t getTagRawValue() const { return m_value; }
    template<typename T>
    T getTagValue() const
    {
        return static_cast<T>(m_value);
    }
    std::bitset<8> getInquiryBits() const { return m_inquiry; }

public: // String getter
    std::string getTagCategoryString() const
    {
        return usereventtag7010types::toString(static_cast<usereventtag7010types::EventCategory>(m_category));
    }
    std::string getTagOccurrenceString() const
    {
        return usereventtag7010types::toString(static_cast<usereventtag7010types::TagOccurrence>(m_occurrence));
    }
    std::string getTagSeverityString() const
    {
        return usereventtag7010types::toString(static_cast<usereventtag7010types::TagSeverity>(m_severity));
    }
    template<typename T>
    std::string getTagValueString() const
    {
        return usereventtag7010types::toString(static_cast<T>(m_value));
    }

public: // Inquiry
    bool hasTagCategory() const { return m_inquiry & inqueryCategory; }
    bool hasTagOccurrence() const { return m_inquiry & inqueryOccurence; }
    bool hasTagSeverity() const { return m_inquiry & inquerySeverity; }
    bool hasTagValue() const { return m_inquiry & inqueryTagValue; }
    bool hasTagUserDefined() const { return m_inquiry & inqueryUserDefined; }

protected:
    NtpTime m_tagStart{}; //!< Timestamp for the start of the event.
    NtpTime m_tagEnd{}; //!< Timestamp for the start of the event.
    uint8_t m_flags{0U}; //!< Indicates the start or end of the event.
    TagClass m_tagClass{TagClass::Unknown}; //!< Defines the classes of the tags.
    uint32_t m_tagId{0U}; //!< The id for the tag.
    std::string m_tagString{}; //!< String to store arbitrary notes, comments or additional information.
    PositionWgs84 m_wgs84{}; //!< Position of the tag.
    uint64_t m_roiWidth{0U}; //!< ROI width for the event around the PositionGPS.
    uint64_t m_roiLength{0U}; //!< ROI length for the event around the PositionGPS.
    ReservedArray m_reserved{{0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U}};

    uint8_t m_category{static_cast<uint8_t>(
        usereventtag7010types::EventCategory::Undefined)}; //!< Top level description of the tag's content
    usereventtag7010types::TagOccurrence m_occurrence{
        usereventtag7010types::TagOccurrence::Undefined}; //!< Appearance of the event
    usereventtag7010types::TagSeverity m_severity{
        usereventtag7010types::TagSeverity::Undefined}; //!< Intensity of the event

    //!The actual event data.
    //!
    //! \note We store it as simple 8 bit unsigned int to allow more event types in future without changing the UserEventTag7010 class.
    uint8_t m_value{0U};
    uint8_t m_inquiry{0U}; //!< Bit mask for querying the tag's content
}; // UserEventTag7010

//==============================================================================
// Comparators
bool operator==(const UserEventTag7010& lhs, const UserEventTag7010& rhs);
bool operator!=(const UserEventTag7010& lhs, const UserEventTag7010& rhs);

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
