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
//! \date Jul 4, 2012
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/io.hpp> //_prototypes.hpp>

#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/noncopyable.hpp>

#include <string>
#include <sstream>
#include <iostream>
#include <iomanip>

#include <time.h>
#include <string.h>
#include <assert.h>

#if defined _WIN32 && defined(MICROVISION_SDKLIBDLL_SHARED)
#    pragma warning(push)
#    pragma warning(disable : 4251)
#endif // _WIN32 && MICROVISION_SDKLIBDLL_SHARED

#ifdef _WIN32
#    if _MSC_VER < 1800 //VS 2013 is not tested 1900 == VS 2015
struct timespec
{
    time_t tv_sec;
    time_t tv_nsec;
};
#    endif //  before VS 2013

using nanoseconds_t = time_t;
#else // _WIN32
using nanoseconds_t = long int;
#endif // _WIN32

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
namespace time {
//==============================================================================

//==============================================================================
//!\brief Return the real-time clock of the system.
//!\return The current UTC time
//!\sa microvision::localTime()
//------------------------------------------------------------------------------
boost::posix_time::ptime universalTime();

//==============================================================================
//!\brief Return the real-time clock of the system.
//!\return The current UTC time
//!\sa microvision::universalTime()
//------------------------------------------------------------------------------
boost::posix_time::ptime localTime();

//==============================================================================
} // namespace time
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

class TimeConversion final : private boost::noncopyable
{
public:
    TimeConversion();
    TimeConversion(const std::string& formatStr);
    ~TimeConversion();

public:
    const char* toString(const timespec ts, const int secPrecision = 0) const;
    const char* toString(const tm& ltime, const nanoseconds_t nanoseconds, const int secPrecision = 0) const;
    const char* toString(const time_t secs) const;
    //	const char* toString(const long int secs, const uint32_t nSecs, const uint32_t nbOfDigits) const;
    std::string toString(const boost::posix_time::ptime ts, const int secPrecision = 0) const;

    std::string toStdString(const time_t secs) const;

protected:
    static const int szDefaultFmt = 18;
    static const char defaultFmt[szDefaultFmt];

    static const int szTimeStr = 64;

protected:
    char* fmt;
    mutable char timeStr[szTimeStr];
}; // TimeConversion

//==============================================================================

class MICROVISION_SDK_API NtpTime final
{
public:
    //========================================
    //! \brief Returns a invalid time.
    //! \return A new Time with invalid stamp.
    //----------------------------------------
    static NtpTime getInvalidTime()
    {
        NtpTime t;
        t.setInvalid();
        return t;
    }

protected:
    static double round(const double v);

public:
    //========================================
    //! \brief Default constructor.
    //----------------------------------------
    NtpTime() noexcept : m_Time(0) {}

    //========================================
    //! \brief Copy constructor for NtpTime.
    //! \param[in] time  The source from which this instance shall be filled.
    //----------------------------------------
    NtpTime(const uint64_t time) noexcept : m_Time(time) {}

    //========================================
    //! \brief Constructor for constructing a time with seconds and fractions of seconds.
    //!        A fractions is a  1/(2^32) of a second.
    //! \param[in] sec   The current Time in seconds.
    //! \param[in] frac  The current fraction of a second.
    //----------------------------------------
    NtpTime(const uint32_t sec, const uint32_t frac) noexcept : m_Time(0) { this->set(sec, frac); }

    //========================================
    //! \brief Constructor for constructing a time with a posix time.
    //! \param[in] timestamp  The posix time from which this instance shall be filled.
    //----------------------------------------
    NtpTime(const boost::posix_time::ptime& timestamp) noexcept;

public: // Assignment operators
    //========================================
    //! \brief Assigns another time to this one.
    //! \param[in] other  The time which shall be assigned to this one.
    //! \return A reference to this after the assignment.
    //----------------------------------------
    NtpTime& operator=(const uint64_t u)
    {
        m_Time = u;
        return *this;
    }

    //========================================
    //! \brief Adds another time to this one.
    //! \param[in] other  The time which shall be added to this one.
    //! \return A reference to this after the calculation.
    //----------------------------------------
    NtpTime& operator+=(const NtpTime& Q)
    {
        m_Time += Q.m_Time;
        return *this;
    }

    //========================================
    //! \brief Subtracts another time from this one.
    //! \param[in] other  The time which shall be subtracted from this one.
    //! \return A reference to this after the calculation.
    //! \note This time will be set invalid if subtracted time is larger than this one!
    //----------------------------------------
    NtpTime& operator-=(const NtpTime& other)
    {
        if (other > *this)
        {
            this->setInvalid();
            return *this;
        }

        m_Time -= other.m_Time;
        return *this;
    }

public: // Cast operators
    //========================================
    //! \brief Casts the time into an uint64_t.
    //! \return The time as uint64_t.
    //----------------------------------------
    operator uint64_t() const { return this->m_Time; }

public: // comparison operators
    //========================================
    //! \brief Checks for equality.
    //! \param[in] other  The time, this time shall be compared to.
    //! \return \c True, if the times are identical, \c false otherwise.
    //----------------------------------------
    bool operator==(const NtpTime& other) const { return (m_Time == other.m_Time); }

    //========================================
    //! \brief Checks for inequality.
    //! \param[in] other  The time, this time shall be compared to.
    //! \return \c True, if the times are not identical, \c false otherwise.
    //----------------------------------------
    bool operator!=(const NtpTime& other) const { return (m_Time != other.m_Time); }

    //========================================
    //! \brief Checks for greater or equality.
    //! \param[in] other  The time, this time shall be compared to.
    //! \return \c True, if the time is greater or equal than the compared, false otherwise.
    //----------------------------------------
    bool operator>=(const NtpTime& other) const { return (m_Time >= other.m_Time); }

    //========================================
    //! \brief Checks for greater.
    //! \param[in] other  The time, this time shall be compared to.
    //! \return \c True, if the time is greater than the compared, false otherwise.
    //----------------------------------------
    bool operator>(const NtpTime& other) const { return (m_Time > other.m_Time); }

    //========================================
    //! \brief Checks for lesser or equality.
    //! \param[in] other  The time, this time shall be compared to.
    //! \return \c True, if the time is lesser or equal than the compared, false otherwise.
    //----------------------------------------
    bool operator<=(const NtpTime& other) const { return (m_Time <= other.m_Time); }

    //========================================
    //! \brief Checks for lesser.
    //! \param[in] other  The time, this time shall be compared to.
    //! \return \c True, if the time is lesser than the compared, false otherwise.
    //----------------------------------------
    bool operator<(const NtpTime& other) const { return (m_Time < other.m_Time); }

public: // arithmetic operators
    //========================================
    //! \brief Adds another time to this one.
    //! \param[in] other  The time which shall be added to this one.
    //! \return A new NtpTime holding the result of the calculation.
    //----------------------------------------
    NtpTime operator+(const NtpTime& other) const
    {
        NtpTime result = *this;
        result += other;
        return result;
    }

    //========================================
    //! \brief Subtracts another time from this one.
    //! \param[in] other  The time which shall be subtracted from this one.
    //! \return A new NtpTime holding the result of the calculation.
    //! \note Result will invalid if subtracted time is larger than this one!
    //----------------------------------------
    NtpTime operator-(const NtpTime& other) const
    {
        if (other > *this)
        {
            return getInvalidTime();
        }

        NtpTime result = *this;
        result -= other;
        return result;
    }

public: //get
    //========================================
    //! \brief Returns the current time in seconds.
    //! \return The current Time in seconds.
    //----------------------------------------
    uint32_t getSeconds() const { return static_cast<uint32_t>(m_Time >> 32); }

    //========================================
    //! \brief Returns the current time in milli seconds.
    //! \return The current Time in milli seconds.
    //! \note Take care for possible overflow and rounding errors.
    //----------------------------------------
    uint32_t getMilliseconds() const
    {
        const uint64_t t = millisecondsToSecondsMultiplicator * m_Time;
        return static_cast<uint32_t>((t >> 32) & 0xFFFFFFFF);
    }

    //========================================
    //! \brief Returns a more precisely time in milli seconds.
    //! \return The current Time in milli seconds.
    //! \note Take care for possible overflow.
    //! 10^3/2^32 = 5^3/2^29 is ca
    //----------------------------------------
    uint32_t getMillisecondsPrecise() const
    {
        // TODO find faster algorithm avoiding double?
        const uint64_t t
            = static_cast<uint64_t>(round(static_cast<double>(m_Time) * secondFractionNtpToNanosecondsFactor * 1e-6));
        return static_cast<uint32_t>(t & 0xFFFFFFFF);
    }

    //========================================
    //! \brief Returns a more precise time in milliseconds.
    //! \return The current time in milliseconds.
    //! \note Take care for possible overflow.
    //! 10^3/2^32 = 5^3/2^29 is ca
    //! \note This method always rounds down!
    //----------------------------------------
    uint32_t getMillisecondsPreciseRoundDown() const
    {
        // TODO find faster algorithm avoiding double?
        const uint64_t t
            = static_cast<uint64_t>(floor(static_cast<double>(m_Time) * secondFractionNtpToNanosecondsFactor * 1e-6));
        return static_cast<uint32_t>(t & 0xFFFFFFFF);
    }

    //========================================
    //! \brief Returns the current time in micro seconds.
    //! \return The current Time in micro seconds.
    //! \note Take care for possible overflow and rounding errors.
    //! 10^6/2^32 = 5^6/2^26 is ca 5^3/536871
    //----------------------------------------
    uint32_t getMicroseconds() const
    {
        const uint64_t t = fractionsToMicrosecondsMultiplicator * m_Time / fractionsToMicrosecondsDivisor;
        return static_cast<uint32_t>(t & 0xFFFFFFFF);
    }

    //========================================
    //! \brief Returns the current time in nano seconds.
    //! \return The current Time in nano seconds.
    //! \note Take care for possible overflow and rounding errors.
    //----------------------------------------
    uint64_t getNanoseconds() const
    {
        // TODO find faster algorithm avoiding double?
        return static_cast<uint64_t>(round(static_cast<double>(m_Time) * secondFractionNtpToNanosecondsFactor));
    }

    //========================================
    //! \brief Returns the fractional part of the second.
    //! \return The fractional part of the seconds.
    //! \note Take care for possible overflow.
    //----------------------------------------
    uint32_t getFracSeconds() const { return static_cast<uint32_t>(m_Time & 0xFFFFFFFF); }

    //========================================
    //! \brief Returns the time in seconds and microseconds (micros: 0..1000 0000). conversion error: 0.. -7.6 us.
    //! \param[out] sec The seconds of the new time.
    //! \param[out] us The microseconds of the new time.
    //!
    //! (u stands for the greek letter Mu).
    //----------------------------------------
    void getTime_s_us(uint32_t& sec, uint32_t& us) const
    {
        sec = getSeconds();
        us  = static_cast<uint32_t>(static_cast<uint64_t>(getFracSeconds()) / fractionsToSecondsDivisor);
    }

    //========================================
    //! \brief Returns the time.
    //! \return The time in seconds and fractional seconds.
    //----------------------------------------
    uint64_t getTime() const { return m_Time; }

public: //set
    //========================================
    //! \brief Sets the time in seconds.
    //! \param[in] u The new Time in seconds.
    //----------------------------------------
    void setSeconds(const uint32_t u) { m_Time = static_cast<uint64_t>(u) << 32; }

    //========================================
    //! \brief Sets the time in milli seconds.
    //! \param[in] u The new Time in milli seconds.
    //----------------------------------------
    void setMilliseconds(const uint32_t u)
    {
        m_Time = static_cast<uint64_t>(u) * millisecondsToFractionsMultiplicator / millisecondsToFractionsDivisor;
    }

    //========================================
    //! \brief Sets the time in micro seconds.
    //! \param[in] u The new Time in micro seconds.
    //!
    //! This routine uses the factorization
    //! 2^32/10^6 = 4096 + 256 - 1825/32
    //----------------------------------------
    void setMicroseconds(const uint32_t u)
    {
        const uint64_t t = (static_cast<uint64_t>(u) * 1825) >> 5;
        m_Time           = (static_cast<uint64_t>(u) << 12) + (static_cast<uint64_t>(u) << 8) - t;
    }

    //========================================
    //! \brief Sets the time in micro seconds.
    //! \param[in] u The new Time in micro seconds.
    //!
    //! This routine uses the factorization
    //! 2^32/10^6 = 4096 + 256 - 1825/32
    //----------------------------------------
    void setMicroseconds(const uint64_t u)
    {
        const uint64_t t = (u * microsecondsToFractionMultiplicator) >> 5;
        m_Time           = (u << 12) + (u << 8) - t;
    }

    //========================================
    //! \brief Sets the time in nano seconds.
    //! \param[in] u The new Time in nano seconds.
    //----------------------------------------
    void setNanoseconds(const uint64_t u)
    {
        // TODO find faster algorithm avoiding double?
        m_Time = static_cast<uint64_t>(round(static_cast<double>(u) * nanosecondsToSecondFractionNtpFactor));
    }

    //========================================
    //! \brief Sets the time in seconds and microseconds (micros: 0..1000 0000). conversion error: 0.. -7.6 us.
    //! \param[in] sec The seconds of the new time.
    //! \param[in] us The microseconds of the new time.
    //!
    //! (u stands for the greek letter Mu).
    //! This routine uses the factorization
    //! 2^32/10^6 = 4096 + 256 - 1825/32
    //----------------------------------------
    void setTime_s_us(const uint32_t sec, const uint32_t us)
    {
        m_Time = (static_cast<uint64_t>(sec) << 32)
                 | ((static_cast<uint64_t>(us) << 12) - ((us * microsecondsToFractionMultiplicator) >> 5)
                    + (static_cast<uint64_t>(us) << 8));
    }

    //========================================
    //! \brief Sets the time.
    //! \param[in] u The new Time.
    //----------------------------------------
    void set(const uint64_t& u) { m_Time = u; }

    //========================================
    //! \brief Sets the time.
    //! \param[in] sec The seconds of the new time.
    //! \param[in] frac The fractions of a second of the new time.
    //----------------------------------------
    void set(const uint32_t sec, const uint32_t frac)
    {
        m_Time = sec;
        m_Time = m_Time << 32;
        m_Time |= frac;
    }

    //========================================
    //! \brief Sets the time to invalid.
    //----------------------------------------
    void setInvalid() { m_Time = uint64_t(NOT_A_DATE_TIME_VALUE) << 32; }

public: //add
    //========================================
    //! \brief Adds seconds to this time.
    //! \param[in] s  The seconds which shall be added to this time.
    //! \return A reference to this after the calculation.
    //----------------------------------------
    void addSeconds(const uint32_t s)
    {
        NtpTime t;
        t.setSeconds(s);
        *this += t;
    }

    //========================================
    //! \brief Adds milli seconds to this time.
    //! \param[in] m  The milli seconds which shall be added to this time.
    //! \return A reference to this after the calculation.
    //----------------------------------------
    void addMilliseconds(const uint32_t m)
    {
        NtpTime t;
        t.setMilliseconds(m);
        *this += t;
    }

    //========================================
    //! \brief Adds micro seconds to this time.
    //! \param[in] u  The micro seconds which shall be added to this time.
    //! \return A reference to this after the calculation.
    //----------------------------------------
    void addMicroseconds(const uint32_t u)
    {
        NtpTime t;
        t.setMicroseconds(u);
        *this += t;
    }

    //========================================
    //! \brief Adds nano seconds to this time.
    //! \param[in] n  The nano seconds which shall be added to this time.
    //! \return A reference to this after the calculation.
    //----------------------------------------
    void addNanoseconds(const uint64_t n)
    {
        NtpTime t;
        t.setNanoseconds(n);
        *this += t;
    }

public: //sub
    //========================================
    //! \brief Subtracts seconds from this time.
    //! \param[in] s  The seconds which shall be subtracted to this time.
    //! \return A reference to this after the calculation.
    //----------------------------------------
    void subSeconds(const uint32_t s)
    {
        NtpTime t;
        t.setSeconds(s);
        *this -= t;
    }

    //========================================
    //! \brief Subtracts milli seconds from this time.
    //! \param[in] s  The milli seconds which shall be subtracted to this time.
    //! \return A reference to this after the calculation.
    //----------------------------------------
    void subMilliseconds(const uint32_t m)
    {
        NtpTime t;
        t.setMilliseconds(m);
        *this -= t;
    }

    //========================================
    //! \brief Subtracts micro seconds from this time.
    //! \param[in] s  The micro seconds which shall be subtracted to this time.
    //! \return A reference to this after the calculation.
    //----------------------------------------
    void subMicroseconds(const uint32_t u)
    {
        NtpTime t;
        t.setMicroseconds(u);
        *this -= t;
    }

    //========================================
    //! \brief Subtracts nano seconds from this time.
    //! \param[in] s  The nano seconds which shall be subtracted to this time.
    //! \return A reference to this after the calculation.
    //----------------------------------------
    void subNanoseconds(const uint64_t n)
    {
        NtpTime t;
        t.setNanoseconds(n);
        *this -= t;
    }

public: // posix time converter
    //========================================
    //! \brief Gets the duration since epoch as posix duration.
    //! \return  The posix duration.
    //!
    //! NtpTime epoch 1-1-1900, Posix epoch 1-1-1970
    //----------------------------------------
    boost::posix_time::time_duration toTimeDurationSinceEpoch() const;

    //========================================
    //! \brief Gets the time as posix time.
    //! \return The posix time. If not a date it returns a boost::posix_time::not_a_date_time
    //----------------------------------------
    boost::posix_time::ptime toPtime() const;

    //========================================
    //! \brief Sets the time from posix time.
    //! \param[in] timestamp  The posix time.
    //----------------------------------------
    void setFromPTime(const boost::posix_time::ptime& timestamp);

public:
    //========================================
    //! \brief Returns if this timestamp does not represent an actual time.
    //! \return \c True if this timestamp does not represent an actual time, else: \c false
    //----------------------------------------
    bool is_not_a_date_time() const;

public:
    //========================================
    //!\brief Constants to convert fractions of a second: 1/(2^32) s (Ntp) to nanoseconds (1e-9 s).
    //!
    //! For efficiency, the Ntp epoch and factors to convert between ns and 1/(2^32)s
    //! are saved in static variables that are computed only once at system initialization.
    //----------------------------------------
    static constexpr double secondFractionNtpToNanosecondsFactor = 0.232830643653869628; // = 2^-32 * 1e9
    static const double secondFractionNtpToNanoseconds; // deprecated - cannot be used outside sdk-core

    //========================================
    //!\brief Constants to convert nanoseconds (1e-9 s) to fractions of a second: 1/(2^32) s (Ntp).
    //!
    //! For efficiency, the Ntp epoch and factors to convert between ns and 1/(2^32)s
    //! are saved in static variables that are computed only once at system initialization.
    //----------------------------------------
    static constexpr double nanosecondsToSecondFractionNtpFactor = 4.294967296000000000; // = 2^32 * 1e-9
    static const double nanosecondsToSecondFractionNtp; // deprecated - cannot be used outside sdk-core

private:
    //!\brief Constants to easy convert milliseconds (1e-3 s) to fractions of a second (1/(2^32) s).
    static const uint64_t millisecondsToFractionsMultiplicator = 536870912;
    static const uint64_t millisecondsToFractionsDivisor       = 125;
    static const uint64_t microsecondsToFractionMultiplicator  = 1825;

    //!\brief Constants to convert fractions of a second (1/(2^32) s) to microseconds (1e-6 s).
    static const uint64_t fractionsToMicrosecondsDivisor       = 536871;
    static const uint64_t fractionsToMicrosecondsMultiplicator = 125;
    static const uint64_t fractionsToSecondsDivisor            = 4295;

    static const uint64_t millisecondsToSecondsMultiplicator = 1000;

    static const uint32_t NOT_A_DATE_TIME; // deprecated - cannot be used outside sdk-core
    static constexpr uint32_t NOT_A_DATE_TIME_VALUE
        = 0xFFFFFFFF; //!< Representation of a not_a_date_time value in the serialization
    static const uint64_t NOT_A_DATE_TIME64; //!< Representation of a not_a_date_time value in the serialization
    static const boost::posix_time::ptime m_epoch; //!< Constant for converting to posix time

protected:
    uint64_t m_Time; //!< Ntp time in 1/2^32 seconds (~233 ps)

public: // friend functions for serialization
    template<typename TT>
    friend void readBE(std::istream& is, TT& value);
    template<typename TT>
    friend TT readBE(std::istream& is);
    template<typename TT>
    friend void readLE(std::istream& is, TT& value);
    template<typename TT>
    friend TT readLE(std::istream& is);

    template<typename TT>
    friend void writeBE(std::ostream& os, const TT& value);
    template<typename TT>
    friend void writeLE(std::ostream& os, const TT& value);
}; // NtpTime

//==============================================================================
//! \brief Read a value of NtpTime in 16 bytes from a stream.
//!        Reading individual bytes is done in big-endian byte order.
//!
//! \param[in, out] is      Stream providing the data to be read.
//! \param[out]     value   On exit it will hold the value that has been read.
//------------------------------------------------------------------------------
void readBE128(std::istream& is, NtpTime& value);

//==============================================================================

//==============================================================================
//! \brief Write a value of NtpTime in 16 bytes into a stream.
//!        Writing individual bytes is done in big-endian byte order.
//!
//! \param[in, out] os      Stream that will receive the data to be written.
//! \param[in]      value   The value to be written.
//------------------------------------------------------------------------------
void writeBE128(std::ostream& os, const NtpTime& value);

//==============================================================================

template<>
inline void readBE<NtpTime>(std::istream& is, NtpTime& value)
{
    microvision::common::sdk::readBE(is, value.m_Time);
}

//==============================================================================

template<>
inline NtpTime readBE<NtpTime>(std::istream& is)
{
    NtpTime t;
    microvision::common::sdk::readBE<NtpTime>(is, t);
    return t;
}

//==============================================================================

template<>
inline void readLE<NtpTime>(std::istream& is, NtpTime& value)
{
    microvision::common::sdk::readLE(is, value.m_Time);
}

//==============================================================================

template<>
inline NtpTime readLE<NtpTime>(std::istream& is)
{
    NtpTime t;
    microvision::common::sdk::readLE(is, t);
    return t;
}

//==============================================================================

template<>
inline void writeBE<NtpTime>(std::ostream& os, const NtpTime& value)
{
    microvision::common::sdk::writeBE(os, value.m_Time);
}

//==============================================================================
template<>
inline void writeLE<NtpTime>(std::ostream& os, const NtpTime& value)
{
    microvision::common::sdk::writeLE(os, value.m_Time);
}

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================

#if defined _WIN32 && defined(MICROVISION_SDKLIBDLL_SHARED)
#    pragma warning(pop)
#endif // _WIN32 && MICROVISION_SDKLIBDLL_SHARED
