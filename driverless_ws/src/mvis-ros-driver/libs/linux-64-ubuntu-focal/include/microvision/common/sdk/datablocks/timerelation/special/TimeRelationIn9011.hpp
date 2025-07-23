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
//! \date March 13, 2018
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/ClockType.hpp>
#include <microvision/common/sdk/NtpTime.hpp>

#include <deque>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

class TimeRelationIn9011 final
{
public:
    //Data stored alongside each Ref time in the EntryDataList
    struct Entry
    {
        Entry();
        Entry(Entry const&);
        Entry& operator=(Entry const&);

    public:
        NtpTime m_refTime;
        NtpTime m_otherTime;
        NtpTime m_standardError;
        NtpTime m_maxDeviation;
        double m_slope;
        double m_correlationCoefficient;
    }; // Entry

    using EntryList = std::deque<Entry>;

public:
    TimeRelationIn9011();
    virtual ~TimeRelationIn9011();

public:
    std::streamsize getSerializedSize() const;
    bool deserialize(std::istream& is);
    bool serialize(std::ostream& os) const;

public: // setter
    void setOtherClock(const ClockType otherClock) { m_otherClock = otherClock; }

    void setEntryList(const EntryList& entryList) { m_entryList = entryList; }

public: // getter
    const ClockType getOtherClock() const { return m_otherClock; }

    const EntryList& getEntryList() const { return m_entryList; }

    EntryList& getEntryList() { return m_entryList; }

protected:
    ClockType m_otherClock;

    // uint32_t; serialization size of the m_entryList
    EntryList m_entryList;
}; // TimeRelationIn9011

//==============================================================================

bool operator==(const TimeRelationIn9011& lhs, const TimeRelationIn9011& rhs);
bool operator!=(const TimeRelationIn9011& lhs, const TimeRelationIn9011& rhs);

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
