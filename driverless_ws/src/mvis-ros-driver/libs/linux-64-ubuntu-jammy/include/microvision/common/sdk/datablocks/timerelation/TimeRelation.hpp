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

#include <microvision/common/sdk/datablocks/TimeRelationBase.hpp>
#include <microvision/common/sdk/datablocks/timerelation/special/TimeRelationIn9010.hpp>
#include <microvision/common/sdk/datablocks/timerelation/special/TimeRelationIn9011.hpp>
#include <microvision/common/sdk/Math.hpp>

#include <boost/scoped_ptr.hpp>
#include <deque>
#include <exception>
#include <memory>
#include <utility>
#include <vector>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief TimeRelation stores information relating the reference timebase to an
//! 'Other' timebase. OfflineTimeRelation allows conversion of times between the
//! reference and the 'Other' timebase.
//!
//! Times in the reference timebase are stored as boost::posix_time::ptime
//! objects, i.e. a specific time e.g. 5 Jan 2017, 15:30:12.000221.
//!
//! Times in the other timebase are stored as boost::posix_time::time_duration
//! objects, i.e. a length of time e.g. 3 minutes, 0 seconds, 213213
//! microseconds. The epoch for the other timebase is not stored, but is the
//! responsibility of the user.
//------------------------------------------------------------------------------
class TimeRelation final
{
public:
    TimeRelation();
    TimeRelation(const TimeRelation& other);
    TimeRelation(const TimeRelationIn9010& other);
    TimeRelation(const TimeRelationIn9011& other);
    TimeRelation& operator  =(const TimeRelation&);
    TimeRelation& operator  =(const TimeRelationIn9010& other);
    TimeRelation& operator  =(const TimeRelationIn9011& other);
    virtual ~TimeRelation() = default;

public:
    struct SmoothingOptionalData
    {
        microvision::common::sdk::timerelation::OtherTime m_standardError;
        microvision::common::sdk::timerelation::OtherTime m_maxDeviation;
        double m_correlationCoefficient;
    };
    // Data stored alongside each Ref time in the EntryDataList
    struct Entry
    {
        Entry();
        Entry(Entry const&);
        Entry& operator=(Entry const&);

        microvision::common::sdk::timerelation::OtherTime
        interpolate(microvision::common::sdk::timerelation::RefTime const& r) const;
        microvision::common::sdk::timerelation::RefTime
        interpolate(microvision::common::sdk::timerelation::OtherTime const& o) const;

    public:
        microvision::common::sdk::timerelation::RefTime m_refTime;
        microvision::common::sdk::timerelation::OtherTime m_otherTime;
        double m_factor;
        // optionalData is empty in old version
        std::unique_ptr<SmoothingOptionalData> m_optionalData;
    };

    using EntryList = std::deque<Entry>;

public: // setter
    void setEntryList(const EntryList& entrylist) { m_entryList = entrylist; }

public: // getter
    const EntryList& getEntryList() const { return m_entryList; }

public: // Accessors
    // ========================================
    //! \brief Check if the OfflineTimeRelation has no information available.
    //!  \return True if there is no information available.
    //!  \throws Never
    //!  If no time-conversion information is available, no time conversions can
    //!  be made.
    // ----------------------------------------
    bool isEmpty() const;

    // ========================================
    //! \brief Check if time-sync ambiguities exist in this OfflineTimeRelation.
    //!  \return True if there is at least one ambiguous OtherTime.
    //!  \throws Never
    //!  If the OfflineTimeRelation is not ambiguous, then no OtherTime has more
    //!  than one equivalent RefTime. However, there may be gaps i.e. some
    //!  OtherTimes that are not in range.
    // ----------------------------------------
    bool isAmbiguous() const;

    // ========================================
    //! \brief Check if there are gaps in the RefTime.
    //!  \return True if there is an out-of-domain RefTime between min and max.
    //!  \throws Never
    // ----------------------------------------
    bool hasRefGap() const;

    // ========================================
    //! \brief Returns the minimum Ref time that can be converted.
    //!  \return The minimum Ref time that can be converted.
    //!  \throws Never
    //!  \sa getRefTimeRanges(), getOtherTimeRanges()
    // ----------------------------------------
    microvision::common::sdk::timerelation::RefTime minRefTime() const;

    // ========================================
    //! \brief Returns the maximum Ref time that can be converted.
    //!  \return The maximum Ref time that can be converted.
    //!  \throws Never
    //!  \sa getRefTimeRanges(), getOtherTimeRanges()
    // ----------------------------------------
    microvision::common::sdk::timerelation::RefTime maxRefTime() const;

    // ========================================
    //! \brief Check if there are gaps in the OtherTime.
    //!  \return True if there is an out-of-domain OtherTime between min and max.
    //!  \throws Never
    // ----------------------------------------
    bool hasOtherGap() const;

    // ========================================
    //! \brief Returns the minimum Other time that can be converted.
    //!  \return The minimum Other time that can be converted.
    //!  \throws Never
    //!  \sa getRefTimeRanges(), getOtherTimeRanges()
    // ----------------------------------------
    microvision::common::sdk::timerelation::OtherTime minOtherTime() const;

    // ========================================
    //! \brief Returns the maximum Other time that can be converted.
    //!  \return The maximum Other time that can be converted.
    //!  \throws Never
    //!  \sa getRefTimeRanges(), getOtherTimeRanges()
    // ----------------------------------------
    microvision::common::sdk::timerelation::OtherTime maxOtherTime() const;

    // ========================================
    //! \brief Returns a vector of the ranges of continuous Ref times.
    //!  \return Vector of the ranges of continuous Ref times.
    //!  \throws Never
    //!  If there is no time-sync information, the vector will be empty.
    //!  If there are jumps i.e. gaps in the data, the vector will have 2+ items.
    //!  Vector items are sorted by time.
    //!  The vector returned by \a getOtherTimeRanges() corresponds to this
    //!  vector.
    // ----------------------------------------
    const microvision::common::sdk::timerelation::RefTimeRangeVector& getRefTimeRanges() const;

    // ========================================
    //! \brief Returns a vector of the ranges of continuous Other times.
    //!  \return Vector of the ranges of continuous Other times.
    //!  \throws Never
    //!  If there is no time-sync information, the vector will be empty.
    //!  If there are jumps i.e. gaps in the data, the vector will have 2+ items.
    //!  The vector returned by \a getRefTimeRanges() corresponds to this vector.
    // ----------------------------------------
    const microvision::common::sdk::timerelation::OtherTimeRangeVector& getOtherTimeRanges() const;

    // ========================================
    //! \brief Returns the non-ambiguous time-range around the specified time.
    //!  \param[in] t Time to find a non-ambiguous time-range.
    //!  \return Non-ambiguous time-range around the specified time.
    //!  \throws OutOfRangeException if the specified Ref time is outside range
    //!  or inside a gap.
    // ----------------------------------------
    const microvision::common::sdk::timerelation::RefTimeRange&
    getUnambiguousRange(microvision::common::sdk::timerelation::RefTime const& t) const;

    // ========================================
    //! \brief Returns the non-ambiguous time-range around the specified time.
    //!  \param[in] t Time to find a non-ambiguous time-range.
    //!  \return Non-ambiguous time-range around the specified time, or invalid
    //!  RefTime values if out-of-range.
    //!  \throws Never
    // ----------------------------------------
    const microvision::common::sdk::timerelation::RefTimeRange&
    getUnambiguousRangeNoThrow(microvision::common::sdk::timerelation::RefTime const& t) const;

    // ========================================
    //! \brief Convert the specified Ref time to the corresponding Other time.
    //!  \param[in] t Ref time to convert.
    //!  \return Time converted to Other timebase.
    //!  \throws OutOfRangeException if the specified Ref time is outside range
    //!  or inside a gap.
    //!  \sa minRefTime(), maxRefTime()
    // ----------------------------------------
    microvision::common::sdk::timerelation::OtherTime
    convert(microvision::common::sdk::timerelation::RefTime const& t) const;

    // ========================================
    //! \brief Convert the specified Ref time to the corresponding Other time.
    //!  \param[in] t Ref time to convert.
    //!  \return Time converted to Other timebase, or invalidOtherTime if out-of-
    //!  range.
    //!  \throws Never
    // ----------------------------------------
    microvision::common::sdk::timerelation::OtherTime
    convertNoThrow(microvision::common::sdk::timerelation::RefTime const& t) const;

    // ========================================
    //! \brief Convert the specified Other time to corresponding Ref time.
    //!  \param[in] t Other time to convert.
    //!  \return Time converted to Ref timebase.
    //!  \throws AmbiguousException if there are multiple valid Ref times for the
    //!  specified Other time.
    //!  \throws OutOfRangeException if the specified Other time is outside range
    //!  or inside a gap.
    //!
    //!  Note: this function will throw exceptions if there are ambiguities.
    //!  \sa convertAll(), convertWithGuess()
    // ----------------------------------------
    microvision::common::sdk::timerelation::RefTime
    convert(microvision::common::sdk::timerelation::OtherTime const& t) const;

    // ========================================
    //! \brief Convert the specified Other time to corresponding Ref time.
    //!  \param[in] t Other time to convert.
    //!  \return Time converted to Ref timebase, or invalidRefTime if out-of-
    //!  range or result is ambiguous.
    //!  \throws Never
    // ----------------------------------------
    microvision::common::sdk::timerelation::RefTime
    convertNoThrow(microvision::common::sdk::timerelation::OtherTime const& t) const;

    // ========================================
    //! \brief Convert the specified Other time to all valid Ref times.
    //!  \param[in] t Other time to convert.
    //!  \return Times converted to Ref timebase, empty if no valid times.
    //!  \throws Never
    // ----------------------------------------
    microvision::common::sdk::timerelation::RefTimeVector
    convertAmbiguous(microvision::common::sdk::timerelation::OtherTime const& t) const;

    static double calculateFactor(const microvision::common::sdk::timerelation::RefTime& r1,
                                  const microvision::common::sdk::timerelation::RefTime& r2,
                                  const microvision::common::sdk::timerelation::OtherTime& o1,
                                  const microvision::common::sdk::timerelation::OtherTime& o2);

    // update metadata
    void updateMetadata() const;

    // set metadata accuracy to false
    void setMetadataInaccurate() const;

public: // debugging functions
    std::string debugToString() const;

    // ========================================
    //! \brief Removes all stored data.
    //!  Note: after calling clear(), isEmpty() will return true.
    // ----------------------------------------
    void clear();

private:
    // ========================================
    //! Find the entry that should be used for interpolation of the specified time
    // ----------------------------------------
    EntryList::const_iterator findInterpolationEntry(const microvision::common::sdk::timerelation::RefTime& r) const;

    // ========================================
    //! Find the entry that should be used for interpolation of the specified time
    // ----------------------------------------
    // 	EntryList::const_iterator findInterpolationEntry(const OtherTime& r) const;

private: // Metadata
    struct Metadata
    {
        Metadata()                = default;
        Metadata(const Metadata&) = delete;
        Metadata& assign(const Metadata& other, const TimeRelation& otherParent, const TimeRelation& parent);

        std::size_t findRangeContaining(const microvision::common::sdk::timerelation::RefTime& t) const;

    public:
        // True if no changes have been made since metadata was updated
        bool m_accurate;

        // True if there is at least one OtherTime that has multiple RefTimes
        bool m_ambiguous;

        // True if there is a gap somewhere in the RefTimes
        bool m_hasRefGap;
        // Minimum and maximum valid RefTimes
        microvision::common::sdk::timerelation::RefTimeRange m_refTimeRange;

        // True if there is a gap somewhere in the OtherTimes
        bool m_hasOtherGap;
        // Minimum and maximum valid OtherTimes
        microvision::common::sdk::timerelation::OtherTimeRange m_otherTimeRange;

        // A vector of the valid time-ranges (can each include multiple entries)
        microvision::common::sdk::timerelation::RefTimeRangeVector m_refRanges;
        microvision::common::sdk::timerelation::OtherTimeRangeVector m_otherRanges;
        // A vector of the first entry corresponding to each time-range
        std::vector<EntryList::const_iterator> m_rangeToEntries;
    };

private:
    struct CompRef
    {
        bool operator()(const Entry& e, microvision::common::sdk::timerelation::RefTime const& t) const
        {
            return e.m_refTime < t;
        }

        bool operator()(microvision::common::sdk::timerelation::RefTime const& t, const Entry& e) const
        {
            return operator()(e, t);
        }
    };

private:
    mutable Metadata m_metadata;

protected:
    // ========================================
    //! \brief A Ref-time-ordered list of Entry items.
    //! Each Entry item may have optional data. If it has no optional data,
    //! it is the end of a time-range.
    //! Note: the final entry in the list must be void.
    // ----------------------------------------
    EntryList m_entryList;

}; // TimeRelation

//==============================================================================

bool operator==(const TimeRelation& lhs, const TimeRelation& rhs);
bool operator!=(const TimeRelation& lhs, const TimeRelation& rhs);
bool operator==(const TimeRelation::Entry& lhs, const TimeRelation::Entry& rhs);

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
