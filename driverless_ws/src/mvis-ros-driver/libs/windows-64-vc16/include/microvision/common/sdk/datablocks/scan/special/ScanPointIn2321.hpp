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
//! \date Jun 26, 2018
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <limits>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Class holding distance and intensity information for an echo received by a single laser of a third party lidar scanner.
//------------------------------------------------------------------------------
class ScanPointIn2321 final
{
    friend bool operator==(const ScanPointIn2321& lhs, const ScanPointIn2321& rhs);

public:
    constexpr static const uint8_t maxIntensity{std::numeric_limits<uint8_t>::max()};

public:
    //========================================
    //! \brief Get the size of the serialization (static version).
    //!
    //! \return Number of bytes used by the serialization of this data class.
    //----------------------------------------
    static std::streamsize getSerializedSize_static();

public:
    //========================================
    //! \brief Constructor.
    //----------------------------------------
    ScanPointIn2321() = default;

    //========================================
    //! \brief Copy-constructor.
    //!
    //! \param[in] src  object to create a copy from.
    //----------------------------------------
    ScanPointIn2321(const ScanPointIn2321& src) = default;

    //========================================
    //! \brief Assignment operator.
    //!
    //! \param[in] src  object to be assigned
    //! \return this
    //----------------------------------------
    ScanPointIn2321& operator=(const ScanPointIn2321& src) = default;

    //========================================
    //! \brief Destructor.
    //----------------------------------------
    virtual ~ScanPointIn2321() = default;

public:
    std::streamsize getSerializedSize() const { return getSerializedSize_static(); }
    bool deserialize(std::istream& is);
    bool serialize(std::ostream& os) const;

public: // getter
    //========================================
    //! \brief Get the distance of this point as reported by the scanner.
    //!
    //! \return distance as reported by the scanner
    //----------------------------------------
    uint16_t getDistance() const { return m_distance; }

    //========================================
    //! \brief Get the intensity of this point as reported by the scanner.
    //!
    //! \return intensity as reported by the scanner
    //----------------------------------------
    uint8_t getIntensity() const { return m_intensity; }

public: // setter
    //========================================
    //! \brief Set the distance of this point as reported by the scanner.
    //!
    //! \return distance as reported by the scanner
    //----------------------------------------
    void setDistance(const uint16_t distance) { m_distance = distance; }

    //========================================
    //! \brief Get the distance of this point as reported by the scanner.
    //!
    //! \return distance as reported by the scanner
    //----------------------------------------
    void setIntensity(const uint8_t intensity) { m_intensity = intensity; }

protected:
    uint16_t m_distance{std::numeric_limits<uint16_t>::max()}; // raw value as received from scanner, needs
        // adjustments according to calibration data
    uint8_t m_intensity{std::numeric_limits<uint8_t>::max()}; // raw value as received from scanner, needs
        // adjustments according to calibration data
}; // ScanPointIn2321

//==============================================================================

//==============================================================================
//! \brief Test scan points for equality.
//!
//! \param[in] lhs  left object to compare
//! \param[in] rhs  right object to compare
//! \return \c True if the objects are equal, \c false otherwise
//------------------------------------------------------------------------------
bool operator==(const ScanPointIn2321& lhs, const ScanPointIn2321& rhs);

//==============================================================================
//! \brief Test scan points for in-equality.
//!
//! \param[in] lhs  left object to compare
//! \param[in] rhs  right object to compare
//! \return \c True if the objects are not equal, \c false otherwise
//------------------------------------------------------------------------------
inline bool operator!=(const ScanPointIn2321& lhs, const ScanPointIn2321& rhs) { return !(lhs == rhs); }

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
