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
//! \date Jun 24, 2019
//------------------------------------------------------------------------------
#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/DataTypeId.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Error infos in MVIS SyncBox status message.
//------------------------------------------------------------------------------
class ErrorIn6320 final
{
    template<class ContainerType, DataTypeId::DataType id>
    friend class Importer;
    template<class ContainerType, DataTypeId::DataType id>
    friend class Exporter;

public:
    //! Mask for checking if an error in the configuration was detected (valid only if error source is System).
    static constexpr uint32_t systemErrorFlagConfiguration{0x00000001U};

    //! Mask for checking if a one millisecond jump was detected (valid only if error source is System).
    static constexpr uint32_t systemErrorFlagOneMsJumpDetected{0x00000002U};

    //! Mask for checking if a ten millisecond jump was detected (valid only if error source is System).
    static constexpr uint32_t systemErrorFlagTenMsJumpDetected{0x00000004U};

    //! Mask for checking if a hundred millisecond jump was detected (valid only if error source is System).
    static constexpr uint32_t systemErrorFlagHunMsJumpDetected{0x00000008U};

    //! Mask for checking if a sync timeout was detected (valid only if error source is PTP).
    static constexpr uint32_t ptpErrorFlagSyncTimeout{0x00000001U};

    //! Mask for checking if an announce timeout was detected (valid only if error source is PTP).
    static constexpr uint32_t ptpErrorFlagAnnounceTimeout{0x00000002U};

    //! Mask for checking if a PPS timeout was detected (valid only if error source is GPS).
    static constexpr uint32_t gpsErrorFlagPpsTimeout{0x00000001U};

    //! Mask for checking if an NMEA timeout was detected (valid only if error source is GPS).
    static constexpr uint32_t gpsErrorFlagNmeaTimeout{0x00000002U};

    //! Mask for checking if the NMEA message could not be parsed (valid only if error source is GPS).
    static constexpr uint32_t gpsErrorFlagParserError{0x00000004U};

    //! Mask for checking if a CRC error was detected (valid only if error source is GPS).
    static constexpr uint32_t gpsErrorFlagCrcError{0x00000008U};

    //! Mask for checking if a receive buffer overflow was detected (valid only if error source is GPS).
    static constexpr uint32_t gpsErrorFlagRxBufferOverflow{0x00000010U};

    //! Mask for checking if a transmit buffer overflow was detected (valid only if error source is GPS).
    static constexpr uint32_t gpsErrorFlagTxBufferOverflow{0x00000020U};

    //! Mask for checking if a receive buffer overflow was detected (valid only if error source is CAN).
    static constexpr uint32_t canErrorFlagRxBufferOverflow{0x00000001U};

    //! Mask for checking if a transmit buffer overflow was detected (valid only if error source is CAN).
    static constexpr uint32_t canErrorFlagTxBufferOverflow{0x00000002U};

    //! Mask for checking if a receive buffer overflow was detected (valid only if error source is CAN FD).
    static constexpr uint32_t canFdErrorFlagRxBufferOverflow{0x00000001U};

    //! Mask for checking if a transmit buffer overflow was detected (valid only if error source is CAN FD).
    static constexpr uint32_t canFdErrorFlagTxBufferOverflow{0x00000002U};

    //! \brief The source of this error.
    enum class ErrorSource : uint8_t
    {
        System  = 0x00U,
        Ptp     = 0x01U,
        Ntp     = 0x02U,
        Gps     = 0x03U,
        Can     = 0x04U,
        CanFd   = 0x05U,
        Flexray = 0x06U,
        Unknown = 0xFFU
    };

public:
    //========================================
    //! \brief Constructor.
    //----------------------------------------
    ErrorIn6320() = default;

    //========================================
    //! \brief Constructor.
    //----------------------------------------
    virtual ~ErrorIn6320() = default;

public:
    //========================================
    //! \brief Gets the source of this error.
    //! \return The error source.
    //----------------------------------------
    ErrorSource getErrorSource() const { return m_errorSource; }

    //========================================
    //! \brief Gets the error flags.
    //! \return The error flags.
    //----------------------------------------
    uint32_t getErrorFlags() const { return m_errorFlags; }

public:
    //========================================
    //! \brief Set the source of this error.
    //! \param[in] errorSource  New error source.
    //----------------------------------------
    void setErrorSource(const ErrorSource errorSource) { m_errorSource = errorSource; }

    //========================================
    //! \brief Set the error flags.
    //! \param[in] errorFlags  New error flags.
    //----------------------------------------
    void setErrorFlags(const uint32_t errorFlags) { m_errorFlags = errorFlags; }

protected:
    ErrorSource m_errorSource{ErrorSource::Unknown};
    uint32_t m_errorFlags{0};
}; // ErrorIn6320

//==============================================================================
//! \brief Test ErrorIn6320 objects for equality.
//!
//! \param[in] lhs  left object to compare
//! \param[in] rhs  right object to compare
//! \return \c True if the objects are equal, \c false otherwise.
//------------------------------------------------------------------------------
inline bool operator==(const ErrorIn6320& lhs, const ErrorIn6320& rhs)
{
    return (lhs.getErrorSource() == rhs.getErrorSource()) && (lhs.getErrorFlags() == rhs.getErrorFlags());
}

//==============================================================================
//! \brief Test ErrorIn6320 objects for in-equality.
//!
//! \param[in] lhs  left object to compare
//! \param[in] rhs  right object to compare
//! \return \c True if the objects are not equal, \c false otherwise.
//------------------------------------------------------------------------------
inline bool operator!=(const ErrorIn6320& lhs, const ErrorIn6320& rhs) { return !(lhs == rhs); }

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
