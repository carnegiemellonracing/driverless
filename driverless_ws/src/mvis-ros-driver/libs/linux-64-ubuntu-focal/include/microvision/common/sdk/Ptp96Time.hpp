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
//! \date Aug 19, 2019
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>
#include <microvision/common/sdk/NtpTime.hpp>
#include <microvision/common/sdk/misc/unit.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \class Ptp96Time
//! \brief Precision Time Protocol V2 stored in 96 bytes
//! IEEE 1588-2008 Precision Clock Synchronization Protocol for Networked Measurement and Control Systems Standard V2
//------------------------------------------------------------------------------
class Ptp96Time final
{
public:
    static constexpr uint64_t bitUsageSeconds   = 0x0000FFFFFFFFFFFFU;
    static constexpr uint8_t serializationShift = 32;

public:
    //========================================
    //! \brief Constructor.
    //----------------------------------------
    Ptp96Time() : m_seconds{0}, m_nanoseconds{0}, m_fractionsOfNanoseconds{0} {};

    //========================================
    //! \brief Constructor.
    //----------------------------------------
    Ptp96Time(const uint64_t sec, const uint32_t nano, const uint16_t frac)
      : m_seconds{sec}, m_nanoseconds{nano}, m_fractionsOfNanoseconds{frac} {};

    //========================================
    //! \brief Destructor.
    //----------------------------------------
    ~Ptp96Time() = default;

public: // operator
    //========================================
    //! \brief Checks for equality.
    //! \param[in] other  The time, this time shall be compared to.
    //! \return \c True, if the times are identical, \c false otherwise.
    //----------------------------------------
    bool operator==(const Ptp96Time& other) const
    {
        return (m_seconds == other.m_seconds) //
               && (m_nanoseconds == other.m_nanoseconds) //
               && (m_fractionsOfNanoseconds == other.m_fractionsOfNanoseconds);
    }

    //========================================
    //! \brief Checks for inequality.
    //! \param[in] other  The time, this time shall be compared to.
    //! \return \c True, if the times are not identical, \c false otherwise.
    //----------------------------------------
    bool operator!=(const Ptp96Time& other) const { return !(*this == other); }

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

    //========================================
    //! \brief Returns the fractional part of a nanosecond of the current time.
    //! \return The fractional part of a nanoseconds of the current time.
    //----------------------------------------
    uint16_t getFractionsOfNanoseconds() const { return m_fractionsOfNanoseconds; }

public: //set
    //========================================
    //! \brief Sets the seconds part of the current time in seconds.
    //! \param[in] sec  The new seconds.
    //----------------------------------------
    void setSeconds(const uint64_t sec) { m_seconds = sec & bitUsageSeconds; }

    //========================================
    //! \brief Sets the nanoseconds part of the current time in nanoseconds.
    //! \param[in] nano  The new nanoseconds.
    //----------------------------------------
    void setNanoseconds(const uint32_t nano) { m_nanoseconds = nano; }

    //========================================
    //! \brief Sets the fractions of a nanosecond of the current time.
    //! \param[in] frac  The new fractions.
    //----------------------------------------
    void setFractionsOfNanoseconds(const uint16_t frac) { m_fractionsOfNanoseconds = frac; }

    //========================================
    //! \brief Sets the time.
    //! \param[in] sec   The seconds of the new time.
    //! \param[in] nano  The nanoseconds of the new time.
    //! \param[in] frac  The fractions of a nanosecond of the new time.
    //----------------------------------------
    void set(const uint64_t sec, const uint32_t nano, const uint16_t frac)
    {
        m_seconds                = sec;
        m_nanoseconds            = nano;
        m_fractionsOfNanoseconds = frac;
    }

public: // time converter
    //========================================
    //! \brief Gets the current time as NtpTime
    //! \return The new NtpTime.
    //----------------------------------------
    NtpTime toNtpTime() const
    {
        //TODO: Only first draft! Check conversion

        const uint32_t ntpSec = static_cast<uint32_t>(epochDifferencePtpToNptInSec + this->getSeconds());
        const uint32_t ntpFrac
            = static_cast<uint32_t>(NtpTime::nanosecondsToSecondFractionNtpFactor * this->getNanoseconds());

        NtpTime ntpTime;
        ntpTime.set(ntpSec, ntpFrac);

        return ntpTime;
    }

    //========================================
    //! \brief Sets the time from NtpTime.
    //! \param[in] timestamp  The time as NtpTime.
    //----------------------------------------
    void setFromNtpTime(const NtpTime& timestamp)
    {
        //TODO: Only first draft! Check conversion

        this->setSeconds(static_cast<uint64_t>(timestamp.getSeconds() - epochDifferencePtpToNptInSec));
        this->setNanoseconds(static_cast<uint32_t>(timestamp.getNanoseconds() % unit::time::nanosecondsPerSecond));
        //this->setFractionsOfNanoseconds(xxx);
    };

public: // friend functions for serialization
    friend void readLE(std::istream& is, Ptp96Time& t);
    friend void writeLE(std::ostream& os, const Ptp96Time& t);

protected:
    uint64_t m_seconds; //!< seconds part of the current time in seconds
    uint32_t m_nanoseconds; //!< nanoseconds part of the current time in nanoseconds
    uint16_t m_fractionsOfNanoseconds; //!< fractions of a nanosecond of the current time in (1/2^16 s)

private:
    //========================================
    //! \brief The epoch difference from Ptp anf Npt in seconds (~70y)
    //----------------------------------------
    static const uint32_t epochDifferencePtpToNptInSec = 2208988800;
}; // Ptp96Time

//==============================================================================
// serialization
//==============================================================================

inline void readLE(std::istream& is, Ptp96Time& t)
{
    microvision::common::sdk::readLE(is, t.m_fractionsOfNanoseconds);
    microvision::common::sdk::readLE(is, t.m_nanoseconds);
    uint16_t secPart1;
    uint32_t secPart2;
    readLE(is, secPart2);
    readLE(is, secPart1);
    t.m_seconds = ((static_cast<uint64_t>(secPart1) << Ptp96Time::serializationShift) + static_cast<uint64_t>(secPart2))
                  & Ptp96Time::bitUsageSeconds;
}

//==============================================================================

inline void writeLE(std::ostream& os, const Ptp96Time& t)
{
    microvision::common::sdk::writeLE(os, t.m_fractionsOfNanoseconds);
    microvision::common::sdk::writeLE(os, t.m_nanoseconds);
    uint16_t secPart1 = static_cast<uint16_t>(t.m_seconds >> Ptp96Time::serializationShift);
    uint32_t secPart2 = static_cast<uint32_t>(t.m_seconds);
    microvision::common::sdk::writeLE(os, secPart2);
    microvision::common::sdk::writeLE(os, secPart1);
}

//==============================================================================

inline constexpr std::streamsize serializedSize(const Ptp96Time&) { return 12; }
//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
