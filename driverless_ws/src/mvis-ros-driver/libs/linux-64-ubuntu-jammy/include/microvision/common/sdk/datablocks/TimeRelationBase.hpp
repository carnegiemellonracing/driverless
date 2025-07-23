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
//! \date May 5, 2017
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/Math.hpp> // Required for isNaN() and INT64

#include <boost/date_time/time_duration.hpp>
#include <boost/date_time/posix_time/time_period.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>

#include <exception>
#include <list>
#include <utility>
#include <vector>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
namespace timerelation {
//==============================================================================

//==============================================================================

using TimeDuration         = boost::posix_time::time_duration;
using RefTime              = boost::posix_time::ptime;
using OtherTime            = boost::posix_time::time_duration;
using RefTimeRange         = std::pair<RefTime, RefTime>;
using OtherTimeRange       = std::pair<OtherTime, OtherTime>;
using RefTimeVector        = std::vector<RefTime>;
using RefTimeRangeVector   = std::vector<RefTimeRange>;
using OtherTimeRangeVector = std::vector<OtherTimeRange>;
using RefTimeRangeList     = std::list<RefTimeRange>;
using OtherTimeRangeList   = std::list<OtherTimeRange>;

//==============================================================================

static constexpr double secsToMs = 1.0E6;

//==============================================================================

class AmbiguousException : public std::runtime_error
{
public:
    AmbiguousException() : std::runtime_error("Ambiguous time conversion.") {}
};

class OutOfRangeException : public std::domain_error
{
public:
    OutOfRangeException() : std::domain_error("Specified time is out-of-range or inside gap.") {}
};

inline TimeDuration emptyTimeDuration() { return boost::posix_time::seconds(0); }

inline TimeDuration abs(const TimeDuration& r) { return (r < emptyTimeDuration()) ? (-r) : r; }

inline RefTime invalidRefTime() { return boost::posix_time::not_a_date_time; }

//==============================================================================
//! \brief Epoch time used to convert a ptime to/from an OtherTime.
//!  \return Epoch as RefTime.
//!  \throws None Does not throw.
// ------------------------------------------------------------------------------
inline RefTime epochRefTime() { return RefTime(boost::gregorian::date(1970, 1, 1)); }

inline OtherTime invalidOtherTime() { return boost::posix_time::not_a_date_time; }

inline const RefTimeRange& invalidRefTimeRange()
{
    // Note: some TimeRelation functions return RefTimeRange by ref, so
    // this function must return by ref, therefore it needs a static variable
    static RefTimeRange result(invalidRefTime(), invalidRefTime());
    return result;
}

inline const OtherTimeRange& invalidOtherTimeRange()
{
    // Note: some TimeRelation functions return OtherTimeRange by ref, so
    // this function must return by ref, therefore it needs a static variable
    static OtherTimeRange result(invalidOtherTime(), invalidOtherTime());
    return result;
}

//==============================================================================
//! \brief Converts the specified floating-point number of seconds to an OtherTime.
//!  \param[in] t Number of seconds to convert.
//!  \return Time represented as an OtherTime.
//!  \throws Never
// ------------------------------------------------------------------------------
inline OtherTime convertFloatToTime(const double t)
{
    if (std::isnan(t))
    {
        return boost::posix_time::not_a_date_time;
    }
    else
    {
        return boost::posix_time::microseconds(static_cast<int64_t>(t * secsToMs));
    }
}

//==============================================================================
//! \brief Converts the specified OtherTime to a floating-point number of seconds.
//!  \param[in] t OtherTime to convert.
//!  \return Time represented as floating-point number of seconds.
//!  \throws Never
// ------------------------------------------------------------------------------
inline double convertTimeToFloat(const OtherTime t)
{
    if (t == invalidOtherTime())
    {
        return microvision::common::sdk::NaN_double;
    }
    else
    {
        return static_cast<double>(t.total_microseconds()) * (1.0 / secsToMs);
    }
}

//==============================================================================
// Arithmetic functions
//==============================================================================

inline TimeDuration operator*(const double d, const TimeDuration& d2)
{
    return boost::posix_time::microseconds(static_cast<int64_t>(
        d * secsToMs * static_cast<double>(d2.ticks()) / static_cast<double>(TimeDuration::ticks_per_second())));
}

//==============================================================================

inline TimeDuration operator*(const TimeDuration& d2, const double d) { return d * d2; }

//==============================================================================

inline double operator/(const TimeDuration& d1, const TimeDuration& d2)
{
    return double(d1.ticks()) / double(d2.ticks());
}

//==============================================================================
//! \brief Utility comparison functor, to permit comparing pairs by their first
//!     items.
// ------------------------------------------------------------------------------
template<class T, class Comp = std::less<T>>
class CompFirst
{
public:
    CompFirst() : m_comp() {}

    CompFirst(Comp const& c) : m_comp(c) {}

    template<class TOther>
    bool operator()(const std::pair<T, TOther>& e, T const& t) const
    {
        return m_comp(e.first, t);
    }

    template<class TOther>
    bool operator()(T const& t, const std::pair<T, TOther>& e) const
    {
        return m_comp(t, e.first);
    }

    template<class TOther>
    bool operator()(const std::pair<T, TOther>& e1, const std::pair<T, TOther>& e2) const
    {
        return m_comp(e1.first, e2.first);
    }

private:
    Comp m_comp;
}; // CompFirst

//==============================================================================
//! \brief Utility comparison functor, to permit comparing pairs by their second
//!     items.
// ------------------------------------------------------------------------------
template<class T, class Comp = std::less<T>>
class CompSecond
{
public:
    CompSecond() : m_comp() {}

    CompSecond(Comp const& c) : m_comp(c) {}

    template<class TOther>
    bool operator()(const std::pair<TOther, T>& e, T const& t) const
    {
        return m_comp(e.second, t);
    }

    template<class TOther>
    bool operator()(T const& t, const std::pair<TOther, T>& e) const
    {
        return m_comp(t, e.second);
    }

    template<class TOther>
    bool operator()(const std::pair<TOther, T>& e1, std::pair<TOther, T>& e2) const
    {
        return m_comp(e1.second, e2.second);
    }

private:
    Comp m_comp;
}; // CompSecond

} // namespace timerelation
} // namespace sdk
} // namespace common
} // namespace microvision
