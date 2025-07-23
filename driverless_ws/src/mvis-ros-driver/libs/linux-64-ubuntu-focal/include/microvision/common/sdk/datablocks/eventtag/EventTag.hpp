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
//! \date Jun 17, 2019
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>
#include <microvision/common/sdk/datablocks/DataContainerBase.hpp>
#include <microvision/common/sdk/misc/SdkHash.hpp>

#include <microvision/common/sdk/datablocks/PositionWgs84.hpp>
#include <microvision/common/sdk/datablocks/eventtag/special/EventTag7000.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Event Tag
//!
//! Special data type:
//! \ref microvision::common::sdk::EventTag7000
//------------------------------------------------------------------------------
class EventTag final : public DataContainerBase
{
    template<class ContainerType, DataTypeId::DataType id>
    friend class Importer;
    template<class ContainerType, DataTypeId::DataType id>
    friend class Exporter;

    friend bool operator==(const EventTag& lhs, const EventTag& rhs);

public:
    //========================================
    //! \brief Container type as string.
    //----------------------------------------
    constexpr static const char* const containerType{"sdk.generalcontainer.eventtag"};

    //========================================
    //! \brief Hash value of this container (static version).
    //----------------------------------------
    static constexpr uint64_t getClassIdHashStatic() { return hash(containerType); }

    //========================================
    //! \brief Hash value of this container.
    //----------------------------------------
    uint64_t getClassIdHash() const override { return getClassIdHashStatic(); }

public:
    //========================================
    //! \brief Constructor.
    //----------------------------------------
    EventTag() = default;

    //========================================
    //! \brief Destructor.
    //----------------------------------------
    ~EventTag() override = default;

public: // getter
    //========================================
    //!\brief Get the tagStart of the event tag.
    //----------------------------------------
    NtpTime getTagStart() const { return m_delegate.getTagStart(); }

    //========================================
    //!\brief Get the tagEnd of the event tag.
    //----------------------------------------
    NtpTime getTagEnd() const { return m_delegate.getTagEnd(); }

    //========================================
    //!\brief Get the flags of the event tag.
    //----------------------------------------
    uint8_t getFlags() const { return m_delegate.getFlags(); }

    //========================================
    //!\brief Get the tagClass of the event tag.
    //----------------------------------------
    EventTag7000::TagClass getTagClass() const { return m_delegate.getTagClass(); }

    //========================================
    //!\brief Get the tagId of the event tag.
    //----------------------------------------
    uint32_t getTagId() const { return m_delegate.getTagId(); }

    //========================================
    //!\brief Get the const tagString of the event tag.
    //----------------------------------------
    const std::string& getTagString() const { return m_delegate.getTagString(); }

    //========================================
    //!\brief Get the tagString of the event tag.
    //----------------------------------------
    std::string& getTagString() { return m_delegate.getTagString(); }

    //========================================
    //!\brief Get the const wgs84 position of the event tag.
    //----------------------------------------
    const PositionWgs84& getWgs84() const { return m_delegate.getWgs84(); }

    //========================================
    //!\brief Get the wgs84 position of the event tag.
    //----------------------------------------
    PositionWgs84& getWgs84() { return m_delegate.getWgs84(); }

    //========================================
    //!\brief Get the roiWidth of the event tag.
    //----------------------------------------
    uint64_t getRoiWidth() const { return m_delegate.getRoiWidth(); }

    //========================================
    //!\brief Get the roiLength of the event tag.
    //----------------------------------------
    uint64_t getRoiLength() const { return m_delegate.getRoiLength(); }

    //========================================
    //!\brief Get the position of the reserved index of the event tag.
    //----------------------------------------
    char getReserved(const uint8_t idx) const { return m_delegate.getReserved(idx); }

    //========================================
    //!\brief Get the reserved array of the event tag.
    //----------------------------------------
    const EventTag7000::ReservedArray& getReserved() const { return m_delegate.getReserved(); }

    //========================================
    //!\brief Check the end of the eventtag.
    //----------------------------------------
    bool isEndOfEvent() const { return m_delegate.isEndOfEvent(); }

    //========================================
    //!\brief Check the pulse of the event tag.
    //----------------------------------------
    bool isPulseEvent() const { return m_delegate.isPulseEvent(); }

public: // setter
    //========================================
    //!\brief Set the tagStart of the event tag.
    //----------------------------------------
    void setTagStart(const NtpTime newTagStart) { m_delegate.setTagStart(newTagStart); }

    //========================================
    //!\brief Set the tagEnd of the event tag.
    //----------------------------------------
    void setTagEnd(const NtpTime newTagEnd) { m_delegate.setTagEnd(newTagEnd); }

    //========================================
    //!\brief Set the flags of the event tag.
    //----------------------------------------
    void setFlags(const uint8_t newFlags) { m_delegate.setFlags(newFlags); }

    //========================================
    //!\brief Set the tagClass of the event tag.
    //----------------------------------------
    void setTagClass(const EventTag7000::TagClass newTagClass) { m_delegate.setTagClass(newTagClass); }

    //========================================
    //!\brief Set the tagId of the event tag.
    //----------------------------------------
    void setTagId(const uint32_t newTagId) { m_delegate.setTagId(newTagId); }

    //========================================
    //!\brief Set the tagString of the event tag.
    //----------------------------------------
    void setTagString(const std::string& newTagString) { m_delegate.setTagString(newTagString); }

    //========================================
    //!\brief Set the wsg84 position of the event tag.
    //----------------------------------------
    void setWgs84(const PositionWgs84& newWgs84) { m_delegate.setWgs84(newWgs84); }

    //========================================
    //!\brief Set the roiWidth of the event tag.
    //----------------------------------------
    void setRoiWidth(const uint64_t newRoiWidth) { m_delegate.setRoiWidth(newRoiWidth); }

    //========================================
    //!\brief Set the roiLength of the event tag.
    //----------------------------------------
    void setRoiLength(const uint64_t newRoiLength) { m_delegate.setRoiLength(newRoiLength); }

    //========================================
    //!\brief Set the endOfEvent Flag of the event tag.
    //----------------------------------------
    void setIsEndOfEvent() { m_delegate.setIsEndOfEvent(); }

    //========================================
    //!\brief Unset the endOfEvent Flag of the event tag.
    //----------------------------------------
    void unsetIsEndOfEvent() { m_delegate.unsetIsEndOfEvent(); }

    //========================================
    //!\brief Set the pulseEvent Flag of the event tag.
    //----------------------------------------
    void setIsPulseEvent() { m_delegate.setIsPulseEvent(); }

    //========================================
    //!\brief Unset the pulseEvent Flag of the event tag.
    //----------------------------------------
    void unsetIsPulseEvent() { m_delegate.unsetIsPulseEvent(); }

protected:
    EventTag7000 m_delegate; // only possible specialization currently

}; // EventTag

//==============================================================================

bool operator==(const EventTag& lhs, const EventTag& rhs);
bool operator!=(const EventTag& lhs, const EventTag& rhs);

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
