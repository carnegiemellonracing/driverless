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
//! \date March 19, 2018
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/ClockType.hpp>
#include <microvision/common/sdk/NtpTime.hpp>

#include <boost/scoped_ptr.hpp>
#include <deque>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

class TimeRelationIn9010 final
{
public:
    struct EntryOptionalData
    {
        // The Other time matching this Ref time
        int64_t m_otherTime;
        // The clock-drift factor
        // Factor = delta(Other time) / delta(Ref time)
        double m_factor;
        // FactorInv = delta(Ref time) / delta(Other time)
        double m_factorInv;
    }; // EntryOptionalData

    struct Entry
    {
        Entry();
        Entry(Entry const&);
        Entry& operator=(Entry const&);

    public:
        NtpTime m_refTime;
        boost::scoped_ptr<EntryOptionalData> m_data;
    }; // Entry

    using EntryList = std::deque<Entry>;

public:
    TimeRelationIn9010();
    virtual ~TimeRelationIn9010();

public:
    std::streamsize getSerializedSize() const;
    bool deserialize(std::istream& is);
    bool serialize(std::ostream& os) const;

public: // setter
    void setEntryList(const EntryList& entryList) { m_entryList = entryList; }

public: // getter
    const EntryList& getEntryList() const { return m_entryList; }

    EntryList& getEntryList() { return m_entryList; }

protected:
    // uint32_t getSerializedSize
    EntryList m_entryList;

}; // TimeRelationIn9010

//==============================================================================

bool operator==(const TimeRelationIn9010& lhs, const TimeRelationIn9010& rhs);
bool operator!=(const TimeRelationIn9010& lhs, const TimeRelationIn9010& rhs);
bool operator==(const TimeRelationIn9010::Entry& lhs, const TimeRelationIn9010::Entry& rhs);
//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
