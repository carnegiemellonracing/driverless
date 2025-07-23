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
//! \date Feb 22, 2021
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/WinCompatibility.hpp>
#include <microvision/common/sdk/NtpTime.hpp>
#include <microvision/common/sdk/misc/unit.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \class Ptp64Time
//! \brief Precision Time Protocol V2 stored in 64 bytes
//! IEEE 1588-2008 Precision Clock Synchronization Protocol for Networked Measurement and Control Systems Standard V2
//------------------------------------------------------------------------------
class Ptp64Time final
{
public:
    //========================================
    //! \brief The epoch difference from Ptp anf Npt in seconds (~70y)
    //----------------------------------------
    static constexpr uint32_t epochDifferencePtpToNptInSeconds{2208988800};

public:
    //========================================
    //! \brief Constructor.
    //----------------------------------------
    Ptp64Time() : m_seconds{0}, m_nanoseconds{0} {};

    //========================================
    //! \brief Construct by time.
    //! \param[in] seconds      The seconds of the new time.
    //! \param[in] nanoseconds  The nanoseconds of the new time.
    //----------------------------------------
    Ptp64Time(const uint32_t seconds, const uint32_t nanoseconds) : m_seconds{seconds}, m_nanoseconds{nanoseconds} {};

    //========================================
    //! \brief Destructor.
    //----------------------------------------
    ~Ptp64Time() = default;

public:
    bool isEpochTime() const { return this->m_seconds == 0 && this->m_nanoseconds == 0; }

public: // operator
    //========================================
    //! \brief Checks for equality.
    //! \param[in] lhs  The time, this time shall be compared to.
    //! \param[in] rhs  The time, this time shall be compared to.
    //! \return \c true, if the times are identical, \c false otherwise.
    //----------------------------------------
    friend bool operator==(const Ptp64Time& lhs, const Ptp64Time& rhs)
    {
        return (lhs.m_seconds == rhs.m_seconds) && (lhs.m_nanoseconds == rhs.m_nanoseconds);
    }

    //========================================
    //! \brief Checks for inequality.
    //! \param[in] lhs  The time, this time shall be compared to.
    //! \param[in] rhs  The time, this time shall be compared to.
    //! \return \c true, if the times are not identical, \c false otherwise.
    //----------------------------------------
    friend bool operator!=(const Ptp64Time& lhs, const Ptp64Time& rhs) { return !(lhs == rhs); }

    //========================================
    //! \brief Checks for less than.
    //! \param[in] lhs  The time, this time shall be compared to.
    //! \param[in] rhs  The time, this time shall be compared to.
    //! \return \c true, if the lhs time is less than rhs, \c false otherwise.
    //----------------------------------------
    friend bool operator<(const Ptp64Time& lhs, const Ptp64Time& rhs)
    {
        return ((lhs.m_seconds < rhs.m_seconds)
                || ((lhs.m_seconds == rhs.m_seconds) && (lhs.m_nanoseconds < rhs.m_nanoseconds)));
    }

    //========================================
    //! \brief Checks for greater than.
    //! \param[in] lhs  The time, this time shall be compared to.
    //! \param[in] rhs  The time, this time shall be compared to.
    //! \return \c true, if the lhs time is greater than rhs, \c false otherwise.
    //----------------------------------------
    friend bool operator>(const Ptp64Time& lhs, const Ptp64Time& rhs)
    {
        return ((lhs.m_seconds > rhs.m_seconds)
                || ((lhs.m_seconds == rhs.m_seconds) && (lhs.m_nanoseconds > rhs.m_nanoseconds)));
    }

    //========================================
    //! \brief Checks for less than or equality.
    //! \param[in] lhs  The time, this time shall be compared to.
    //! \param[in] rhs  The time, this time shall be compared to.
    //! \return \c true, if the lhs time is less than or equals rhs, \c false otherwise.
    //----------------------------------------
    friend bool operator<=(const Ptp64Time& lhs, const Ptp64Time& rhs)
    {
        return ((lhs.m_seconds < rhs.m_seconds)
                || ((lhs.m_seconds == rhs.m_seconds) && (lhs.m_nanoseconds <= rhs.m_nanoseconds)));
    }

    //========================================
    //! \brief Checks for greater than or equality.
    //! \param[in] lhs  The time, this time shall be compared to.
    //! \param[in] rhs  The time, this time shall be compared to.
    //! \return \c true, if the lhs time is greater than or equals rhs, \c false otherwise.
    //----------------------------------------
    friend bool operator>=(const Ptp64Time& lhs, const Ptp64Time& rhs)
    {
        return ((lhs.m_seconds > rhs.m_seconds)
                || ((lhs.m_seconds == rhs.m_seconds) && (lhs.m_nanoseconds >= rhs.m_nanoseconds)));
    }

    //========================================
    //! \brief Subtract time from time.
    //! \param[in] lhs  Time to subtract from.
    //! \param[in] rhs  Time to subtract.
    //! \return Duration between lhs and rhs time.
    //----------------------------------------
    friend Ptp64Time operator-(const Ptp64Time& lhs, const Ptp64Time& rhs)
    {
        return Ptp64Time{lhs.m_seconds - rhs.m_seconds, lhs.m_nanoseconds - rhs.m_nanoseconds};
    }

    //========================================
    //! \brief Addition time on time.
    //! \param[in] lhs  Time to addition on.
    //! \param[in] rhs  Time to addition.
    //! \return Duration over lhs and rhs time.
    //----------------------------------------
    friend Ptp64Time operator+(const Ptp64Time& lhs, const Ptp64Time& rhs)
    {
        return Ptp64Time{lhs.m_seconds + rhs.m_seconds, lhs.m_nanoseconds + rhs.m_nanoseconds};
    }

    //========================================
    //! \brief Print human readable formated time.
    //! \param[in, out] outputStream    Output stream to print
    //! \param[in]      time            Time to print
    //! \return Output stream for operator chaining.
    //----------------------------------------
    friend std::ostream& operator<<(std::ostream& outputStream, const Ptp64Time& time)
    {
        return (outputStream << time.toNtpTime().toPtime());
    }

public: //get
    //========================================
    //! \brief Returns the seconds part of the current time in seconds.
    //! \return The seconds part of the current in seconds.
    //----------------------------------------
    uint64_t getSeconds() const { return m_seconds; }

    //========================================
    //! \brief Returns the nanoseconds part of the current time in nanoseconds.
    //! \return The nanoseconds part of the current Time in nanoseconds.
    //----------------------------------------
    uint32_t getNanoseconds() const { return m_nanoseconds; }

public: //set
    //========================================
    //! \brief Sets the seconds part of the current time in seconds.
    //! \param[in] sec  The new seconds.
    //----------------------------------------
    void setSeconds(const uint32_t seconds) { m_seconds = seconds; }

    //========================================
    //! \brief Sets the nanoseconds part of the current time in nanoseconds.
    //! \param[in] nano  The new nanoseconds.
    //----------------------------------------
    void setNanoseconds(const uint32_t nanoseconds) { m_nanoseconds = nanoseconds; }

    //========================================
    //! \brief Set time
    //! \param[in] seconds      The seconds of the new time.
    //! \param[in] nanoseconds  The nanoseconds of the new time.
    //----------------------------------------
    void set(const uint32_t seconds, const uint32_t nanoseconds)
    {
        m_seconds     = seconds;
        m_nanoseconds = nanoseconds;
    }

public: // time converter
    //========================================
    //! \brief Gets the the current time as NtpTime
    //! \return The new NtpTime.
    //----------------------------------------
    NtpTime toNtpTime() const
    {
        //TODO: Only first draft! Check conversion
        NtpTime ntpTime{};
        ntpTime.setSeconds(static_cast<uint32_t>(epochDifferencePtpToNptInSeconds + this->getSeconds()));
        ntpTime.addNanoseconds(this->getNanoseconds());
        return ntpTime;
    }

    //========================================
    //! \brief Sets the the time from NtpTime.
    //! \param[in] timestamp  The time as NtpTime.
    //----------------------------------------
    void setFromNtpTime(const NtpTime& timestamp)
    {
        //TODO: Only first draft! Check conversion
        this->setSeconds(static_cast<uint32_t>(timestamp.getSeconds() - epochDifferencePtpToNptInSeconds));
        this->setNanoseconds(static_cast<uint32_t>(timestamp.getNanoseconds() % unit::time::nanosecondsPerSecond));
    };

private:
    uint32_t m_seconds; //!< seconds part of the current time in seconds
    uint32_t m_nanoseconds; //!< nanoseconds part of the current time in nanoseconds

}; // Ptp64Time

//==============================================================================
// serialization
//==============================================================================

template<>
inline void readLE<Ptp64Time>(std::istream& is, Ptp64Time& time)
{
    time.setSeconds(readLE<uint32_t>(is));
    time.setNanoseconds(readLE<uint32_t>(is));
}

//==============================================================================

template<>
inline Ptp64Time readLE<Ptp64Time>(std::istream& is)
{
    Ptp64Time time;
    readLE(is, time);
    return time;
}

//==============================================================================

template<>
inline void writeLE<Ptp64Time>(std::ostream& os, const Ptp64Time& time)
{
    writeLE(os, static_cast<uint32_t>(time.getSeconds()));
    writeLE(os, static_cast<uint32_t>(time.getNanoseconds()));
}

//==============================================================================

inline constexpr std::streamsize serializedSize(const Ptp64Time&)
{
    return 4 + // bytes for seconds
           4; // bytes for nanoseconds
}

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
