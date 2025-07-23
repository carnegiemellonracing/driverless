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
//! \date Feb 21, 2017
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/NtpTime.hpp>
#include <microvision/common/sdk/Math.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

class ClockType final
{
public:
    enum class ClockName : uint8_t
    {
        Unknown      = 0, //!< The Clock is Unknown.
        Laserscanner = 1, //!< Laser scanner clock is used.
        Ecu          = 2, //!< ECU Clock is used.
        CanBus       = 3, //!< CanBus Clock is used.
        Camera       = 4, //!< Camera Clock is used.
        GpsImu       = 5, //!< GpsImu Clock is used.
        Dut          = 6, //!< DUT Clock is used.
        SyncBox      = 7, //!< SyncBox Clock is used.
        Other        = 255 //!< Another Clock is used.
    };

    static constexpr uint8_t unknownId = 0xFF;

public:
    ClockType();
    ClockType(const uint8_t clockId, const ClockName clockName);
    virtual ~ClockType();

public:
    std::streamsize getSerializedSize() const;
    bool deserialize(std::istream& is);
    bool serialize(std::ostream& os) const;

public: //getter
    uint8_t getClockId() const { return m_clockId; }
    ClockName getClockName() const { return m_clockName; }
    std::string getClockNameString(const ClockName cn);
    bool operator<(const ClockType& other) const;

public: //setter
    void setClockId(const uint8_t clockId) { this->m_clockId = clockId; }
    void setClockName(const ClockName clockName) { this->m_clockName = clockName; }

protected:
    //========================================
    //! \brief A unique ID given to each processing device.
    //!
    //! The ID is required to distinguish between two similar
    //! clock names but from different devices and having
    //! different properties.
    //----------------------------------------
    uint8_t m_clockId;

    //========================================
    //! \brief Name of the clock (enum) indicates the type
    //!        of device it is being received.
    //----------------------------------------
    ClockName m_clockName;
}; // ClockType

//==============================================================================

bool operator==(const ClockType& clk1, const ClockType& clk2);

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
