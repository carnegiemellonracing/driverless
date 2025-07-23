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
//! \date Jun 26, 2019
//! \brief Position in UTM Coordinates
//------------------------------------------------------------------------------

#pragma once

#include <microvision/common/sdk/misc/defines/defines.hpp>
#include <microvision/common/sdk/Math.hpp>
#include <microvision/common/sdk/io.hpp>
#include <microvision/common/sdk/datablocks/PositionWgs84.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \class PositionUtm
//! \brief Position class for which can store a Utm Coordinate.
// ------------------------------------------------------------------------------
class PositionUtm final
{
public:
    //========================================
    //! \brief Default constructor.
    //! Initializes the Utm
    //----------------------------------------
    PositionUtm() = default;

    //========================================
    //! \brief Entry wise constructor
    //----------------------------------------
    PositionUtm(const uint8_t zone, const char band, const double north, const double east, const double mamsl)
      : m_zone{zone}, m_band{band}, m_north{north}, m_east{east}, m_metresAboveMeanSeaLevel{mamsl}
    {}

    //========================================
    //! \brief Constructor with initialisation from WGS84 coordinate.
    //! \param[in] wgsCoord  The UTM coordinate will be initialized by this parameter.
    //----------------------------------------
    explicit PositionUtm(const PositionWgs84& wgsCoord) { setFromPositionWgs84(wgsCoord); }

    //========================================
    //! \brief Destructor.
    //----------------------------------------
    virtual ~PositionUtm() = default;

public:
    bool operator==(const PositionUtm& other) const
    {
        return (m_zone == other.getZone()) //
               && (m_band == other.getBand()) //
               && (fuzzyDoubleEqualT<7>(m_north, other.getNorth())) //
               && (fuzzyDoubleEqualT<7>(m_east, other.getEast())) //
               && (fuzzyDoubleEqualT<7>(m_metresAboveMeanSeaLevel, other.getMetresAboveMeanSeaLevel())) //
            ;
    }

    bool operator!=(const PositionUtm& other) const { return !(*this == other); }

public:
    //========================================
    //!\brief Get the zone of the coordinate.
    //!\return Zone of the coordinate.
    //----------------------------------------
    uint8_t getZone() const { return m_zone; }

    //========================================
    //!\brief Get the band of the coordinate.
    //!\return Band of the coordinate.
    //----------------------------------------
    char getBand() const { return m_band; }

    //========================================
    //!\brief Get the north value of the coordinate.
    //!\return North of the coordinate.
    //----------------------------------------
    double getNorth() const { return m_north; }

    //========================================
    //!\brief Get the east value of the coordinate.
    //!\return East of the coordinate.
    //----------------------------------------
    double getEast() const { return m_east; }

    //========================================
    //!\brief Get the metres above mean sea level in m.
    //!\return Metres above mean sea level of the coordinate.
    //----------------------------------------
    double getMetresAboveMeanSeaLevel() const { return m_metresAboveMeanSeaLevel; }

public:
    //========================================
    //!\brief Set the zone of the coordinate.
    //!\param[in] zone  New zone of the coordinate.
    //----------------------------------------
    void setZone(const uint8_t zone) { m_zone = zone; }

    //========================================
    //!\brief Set the band of the coordinate.
    //!\param[in] band  New band of the coordinate. Capital letter [A-Z]
    //----------------------------------------
    void setBand(const char band) { m_band = band; }

    //========================================
    //!\brief Set the north of the coordinate.
    //!\param[in] north  New north of the coordinate.
    //----------------------------------------
    void setNorth(const double north) { m_north = north; }

    //========================================
    //!\brief Set the east of the coordinate.
    //!\param[in] east  New north of the coordinate.
    //----------------------------------------
    void setEast(const double east) { m_east = east; }

    //========================================
    //!\brief Set the metres above mean sea level in m.
    //!\param[in] mamsl  Difference from the mean sea level.
    //----------------------------------------
    void setMetresAboveMeanSeaLevel(const double mamsl) { m_metresAboveMeanSeaLevel = mamsl; }

    //========================================
    //! \brief Calculate the meridian convergence angle of a given coordinate.
    //!
    //! This angle is the difference between the local UTM north and the true north.
    //! \param[in] wgsCoord  WGS84-coordinate.
    //! \return Convergence angle [rad].
    //----------------------------------------
    double getMeridianConvergenceAngle(const PositionWgs84& wgsCoord) const;

public:
    //========================================
    //! \brief Set UTM position from a given WGS84 coordinates.
    //! \param[in] wgs84  Convert this WGS84 coordinate to UTM.
    //----------------------------------------
    void setFromPositionWgs84(const PositionWgs84& wgs84);

    //========================================
    //! \brief Set UTM position from a given WGS84 coordinates. Enforces a given UTM zone and band.
    //! \param[in] wgs84  Convert this WGS84 coordinate to UTM.
    //! \param[in] zone   Use this UTM zone.
    //! \param[in] band   Use this UTM band. Capital letter [A-Z].
    //----------------------------------------
    void setFromPositionWgs84(const PositionWgs84& wgs84, const uint8_t zone, const char band);

    //========================================
    //! \brief Convert this UTM coordinate to WGS84 coordinate.
    //! \return Converted WGS84 coordinate.
    //----------------------------------------
    PositionWgs84 convertToPositionWgs84() const;

public:
    //========================================
    //!\brief Get the UTM-zone, in which a given WGS-coordinate lays.
    //!\param[in] wgsCoord  WGS84-coordinate, used to determine the zone.
    //!\return UTM-Zone, in which the coordinate lays.
    //----------------------------------------
    static uint8_t getUtmZone(const PositionWgs84& wgsCoord);

    //========================================
    //! \brief Get the UTM-band of the WGS coordinate.
    //! \param[in] wgsCoord  WGS84-coordinate, used to determine the band.
    //! \return UTM-band [A-Z] of the input coordinate.
    //----------------------------------------
    static char getUtmBand(const PositionWgs84& wgsCoord);

    //========================================
    //! \brief Calculate the long-coordinate of the reference meridian, used in a given UTM-Zone.
    //! \param[in] zone  Zone, for which the reference meridian wil be calculated.
    //! \return Longitude [rad] of the reference meridian.
    //----------------------------------------
    static double getUtmRefMeridian(const uint8_t zone);

    //========================================
    //! \brief Check if UTM-band is within the northern hemisphere.
    //! \param[in] band  UTM-band to be checked, as a Capital letter [A-Z].
    //! \return True, if band is within northern hemisphere.
    //----------------------------------------
    static bool isNorthernHemisphere(const char band);

public: // friend functions for serialization
    friend void readBE(std::istream& is, PositionUtm& positionUtm);
    friend void writeBE(std::ostream& os, const PositionUtm& positionUtm);

protected:
    uint8_t m_zone{0}; //<! The zone of the utm coordinate.
    char m_band{}; //<! The band of the utm coordinate.
    double m_north{NaN_double}; //!< The north value of the utm coordinate.
    double m_east{NaN_double}; //!< The east value of the utm coordinate.
    double m_metresAboveMeanSeaLevel{NaN_double}; //!< The metres above mean sea level in [m].

protected:
    // Parameters, approximating the shape of the earth for converting between UTM and longitude/latitude coordinates.
    // Approximation is done by the Krueger series. For more information about the parameters and the Krueger series, see:
    // - Krueger, L. (1912). "Konforme Abbildung des Erdellipsoids in der Ebene"
    // - Karney, Charles F. F. (2011). "Transverse Mercator with an accuracy of a few nanometers"
    // - https://en.wikipedia.org/wiki/Universal_Transverse_Mercator_coordinate_system#Simplified_formulae
    static constexpr double a{6378137.0}; //!< The equatorial radius [m].
    static constexpr double flattening{1.0 / 298.257223563}; //!< The earth ellipsoid flattening.
    static constexpr double N0North{0.0}; //!< The northing offset for northern hemisphere.
    static constexpr double N0South{10000000.0}; //!< The northing offset for southern hemisphere.
    static constexpr double k0{0.9996}; //!< The point scale factor at origin.
    static constexpr double E0{500000.0}; //! The easting offset [m].
    static constexpr double n{flattening / (2.0 - flattening)}; //!< Krueger series parameter n.
    static constexpr double nPow2{n * n}; //!< Krueger series parameter n to the power of 2.
    static constexpr double nPow3{n * n * n}; //!< Krueger series parameter n to the power of 3.
    static constexpr double nPow4{nPow2 * nPow2}; //!< Krueger series parameter n to the power of 4.
    static constexpr double nPow6{nPow4 * nPow2}; //!< Krueger series parameter n to the power of 6.
    static constexpr double nPow8{nPow4 * nPow4}; //!< Krueger series parameter n to the power of 8.
    static constexpr double nPow10{nPow8 * nPow2}; //!< Krueger series parameter n to the power of 10.
    static constexpr double A{(a / (1.0 + n))
                              * (1.0 + (nPow2 / 4) + (nPow4 / 64) + (nPow6 / 256) + (25.0 * nPow8 / 16384)
                                 + (49.0 * nPow10 / 65536))}; //!< Krueger series parameter A.
    static constexpr double alpha1{(0.5 * n) - ((2.0 / 3.0) * nPow2)
                                   + ((5.0 / 16.0) * nPow3)}; //!< Krueger series parameter alpha.
    static constexpr double alpha2{((13.0 / 48.0) * nPow2) - ((3.0 / 5.0) * nPow3)}; //!< Krueger series parameter alpha
    static constexpr double alpha3{(61.0 / 240.0) * nPow3}; //!< Krueger series parameter alpha.
    static constexpr double beta1{(0.5 * n) - ((2.0 / 3.0) * nPow2)}; //!< Krueger series parameter beta.
    static constexpr double beta2{((1.0 / 48.0) * nPow2) + ((1.0 / 15.0) * nPow3)}; //!< Krueger series parameter beta.
    static constexpr double beta3{((17.0 / 480.0)) * nPow3}; //!< Krueger series parameter beta.
    static constexpr double delta1{(2.0 * n) - ((2.0 / 3.0) * nPow2)
                                   + (2.0 * nPow3)}; //!< Krueger series parameter delta.
    static constexpr double delta2{((7.0 / 3.0) * nPow2) - ((8.0 / 5.0) * nPow3)}; //!< Krueger series parameter delta.
    static constexpr double delta3{((56.0 / 15.0) * nPow3)}; //!< Krueger series parameter delta.

    static constexpr double utmZoneWidth{6.0 * deg2rad}; //!< The width of one utm zone [rad].
    static constexpr double utmZoneMeridianOffset{-183.0
                                                  * deg2rad}; //!< The offset, used when calculating a utm zone [rad].
    //! An array of all regular UTM bands [C-X].
    static constexpr std::array<char, 20> utmBands{'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M',
                                                   'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X'};
}; // PositionUtm

//==============================================================================
// specializations
//==============================================================================

inline constexpr std::streamsize serializedSize(const PositionUtm&)
{
    return sizeof(uint8_t) + sizeof(char) + sizeof(double) + sizeof(double) + sizeof(double);
}

inline void readBE(std::istream& is, PositionUtm& positionUtm)
{
    microvision::common::sdk::readBE(is, positionUtm.m_zone);
    microvision::common::sdk::readBE(is, positionUtm.m_band);
    microvision::common::sdk::readBE(is, positionUtm.m_north);
    microvision::common::sdk::readBE(is, positionUtm.m_east);
    microvision::common::sdk::readBE(is, positionUtm.m_metresAboveMeanSeaLevel);
}

inline void writeBE(std::ostream& os, const PositionUtm& positionUtm)
{
    microvision::common::sdk::writeBE(os, positionUtm.m_zone);
    microvision::common::sdk::writeBE(os, positionUtm.m_band);
    microvision::common::sdk::writeBE(os, positionUtm.m_north);
    microvision::common::sdk::writeBE(os, positionUtm.m_east);
    microvision::common::sdk::writeBE(os, positionUtm.m_metresAboveMeanSeaLevel);
}

//==============================================================================

} // namespace sdk
} // namespace common
} // namespace microvision

//==============================================================================
