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
//! \date Nov 25, 2019
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>
#include <microvision/common/sdk/Vector3.hpp>
#include <microvision/common/sdk/Matrix6x6.hpp>
#include <microvision/common/sdk/ScannerType.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Scanner information in the format of the MOVIA scan.
//------------------------------------------------------------------------------
class ScannerInfoIn2340 final
{
public:
    //========================================
    //! \brief Flag to identify in which coordinate system the point positions are.
    //----------------------------------------
    enum class CoordinateSystem : uint8_t
    {
        WorldReference             = 0x00U, //!< World reference coordinate system.
        VehicleRoad                = 0x01U, //!< Vehicle road coordinate system.
        VehicleBody                = 0x02U, //!< Vehicle body coordinate system.
        SensorHousing              = 0x03U, //!< Sensor housing coordinate system.
        SensorMeasurement          = 0x04U, //!< Sensor measurement coordinate system.
        SensorAdjustment           = 0x05U, //!< Sensor adjustment coordinate system.
        IdealizedSensorMeasurement = 0x06U, //!< Idealized sensor measurement coordinate system.
    };

public:
    //========================================
    //! \brief Default constructor.
    //----------------------------------------
    ScannerInfoIn2340() = default;

    //========================================
    //! \brief Default copy-constructor.
    //!
    //! \param[in] src  Object to create a copy from.
    //----------------------------------------
    ScannerInfoIn2340(const ScannerInfoIn2340& src) = default;

    //========================================
    //! \brief Default assignment operator.
    //!
    //! \param[in] src  Object to be assigned
    //! \return this
    //----------------------------------------
    ScannerInfoIn2340& operator=(const ScannerInfoIn2340& src) = default;

public: // getter
    //========================================
    //! \brief Get the unique hardware identifier within sensor network.
    //!
    //! \return The unique hardware identifier.
    //----------------------------------------
    uint64_t getDeviceId() const { return m_deviceId; }

    //========================================
    //! \brief Get the type of scanner.
    //!
    //! Type of scanner which used for originally measurement.
    //!
    //! \return The scanner type.
    //----------------------------------------
    ScannerType getScannerType() const { return m_scannerType; }

    //========================================
    //! \brief Get the coordinate system in which the point positions are.
    //!
    //! \return The coordinate system.
    //----------------------------------------
    CoordinateSystem getCoordinateSystem() const { return m_coordinateSystem; }

    //========================================
    //! \brief Get the horizontal field of view of the scanner in rad.
    //!
    //! The horizontal field of view is positioned forward symmetrically
    //! to the x axis left and right.
    //!
    //! \return The horizontal field of view.
    //----------------------------------------
    double getHorizontalFieldOfView() const { return m_horizontalFieldOfView; }

    //========================================
    //! \brief Get the vertical field of view of the scanner in rad.
    //!
    //! The vertical field of view is positioned forward symmetrically
    //! to the x axis upwards and downwards.
    //!
    //! \return The vertical field of view.
    //----------------------------------------
    double getVerticalFieldOfView() const { return m_verticalFieldOfView; }

    //========================================
    //! \brief Get the mounting position of the scanner in the three dimensions x, y and z in m.
    //!
    //! \return The mounting position.
    //----------------------------------------
    const Vector3<double>& getMountingPosition() const { return m_mountingPosition; }

    //========================================
    //! \brief Get the mounting orientation of the scanner.
    //!
    //! The mounting position given as Euler angles: (roll, pitch, yaw).
    //!
    //! The application order is RPY (matrix notation), i.e. first yaw, then pitch, then roll rotation.
    //!
    //! \return The mounting orientation.
    //----------------------------------------
    const Vector3<double>& getMountingOrientation() const { return m_mountingOrientation; }

    //========================================
    //! \brief Get the horizontal angular resolution of the scanner in rad.
    //!
    //! \return The horizontal angular resolution.
    //----------------------------------------
    double getHorizontalResolution() const { return m_horizontalResolution; }

    //========================================
    //! \brief Get the vertical angular resolution of the scanner in rad.
    //!
    //! \return The vertical angular resolution.
    //----------------------------------------
    double getVerticalResolution() const { return m_verticalResolution; }

    //========================================
    //! \brief Get the uncertainty of the mounting position and orientation
    //! for the six dimensions x, y, z, roll, pitch and yaw.
    //!
    //! \return The uncertainty of the mounting position and orientation.
    //----------------------------------------
    const Matrix6x6<float>& getMountingPositionOrientationSigma() const { return m_mountingPositionOrientationSigma; }

public: // setter
    //========================================
    //! \brief Set the unique hardware identifier within sensor network.
    //!
    //! \param[in] deviceId  The new unique hardware identifier.
    //----------------------------------------
    void setDeviceId(const uint64_t deviceId) { m_deviceId = deviceId; }

    //========================================
    //! \brief Set the type of scanner.
    //!
    //! \param[in] scannerType  The new scanner type.
    //----------------------------------------
    void setScannerType(const ScannerType scannerType) { m_scannerType = scannerType; }

    //========================================
    //! \brief Set the coordinate system of measured positions.
    //!
    //! \param[in] coordinateSystem  The new coordinate system.
    //----------------------------------------
    void setCoordinateSystem(const CoordinateSystem coordinateSystem) { m_coordinateSystem = coordinateSystem; }

    //========================================
    //! \brief Set the horizontal field of view of the scanner in rad.
    //!
    //! \param[in] horizontalFieldOfView  The new horizontal field of view.
    //----------------------------------------
    void setHorizontalFieldOfView(const double horizontalFieldOfView)
    {
        m_horizontalFieldOfView = horizontalFieldOfView;
    }

    //========================================
    //! \brief Set the vertical field of view of the scanner in rad.
    //!
    //! \param[in] verticalFieldOfView  The new vertical field of view.
    //----------------------------------------
    void setVerticalFieldOfView(const double verticalFieldOfView) { m_verticalFieldOfView = verticalFieldOfView; }

    //========================================
    //! \brief Set the mounting position of the scanner in the three dimensions x, y and z in m.
    //!
    //! \param[in] mountingPosition  The new mounting position of the scanner.
    //----------------------------------------
    void setMountingPosition(const Vector3<double>& mountingPosition) { m_mountingPosition = mountingPosition; }

    //========================================
    //! \brief Set the mounting orientation of the scanner in the three dimensions.
    //!
    //! The order in the vector is (roll, pitch, yaw).
    //!
    //! \param[in] mountingOrientation  The new mounting orientation of the scanner.
    //----------------------------------------
    void setMountingOrientation(const Vector3<double>& mountingOrientation)
    {
        m_mountingOrientation = mountingOrientation;
    }

    //========================================
    //! \brief Set the horizontal angular resolution of the scanner in rad.
    //!
    //! \param[in] horizontalResolution  The new horizontal resolution.
    //----------------------------------------
    void setHorizontalResolution(const double horizontalResolution) { m_horizontalResolution = horizontalResolution; }

    //========================================
    //! \brief Set the vertical angular resolution of the scanner in rad.
    //!
    //! \param[in] verticalResolution  The new vertical resolution.
    //----------------------------------------
    void setVerticalResolution(const double verticalResolution) { m_verticalResolution = verticalResolution; }

    //========================================
    //! \brief Set the uncertainty matrix of the mounting position and orientation
    //! for the six dimensions x, y, z, roll, pitch and yaw.
    //!
    //! \param[in] sigma  The new uncertainty matrix.
    //----------------------------------------
    void setMountingPositionOrientationSigma(const Matrix6x6<float>& sigma)
    {
        m_mountingPositionOrientationSigma = sigma;
    }

private:
    uint64_t m_deviceId{0}; //!< Unique hardware identifier within sensor network.
    ScannerType m_scannerType{ScannerType::Movia}; //!< Scanner device type.

    //========================================
    //!\brief Coordinate system of measured positions.
    //----------------------------------------
    CoordinateSystem m_coordinateSystem{CoordinateSystem::SensorMeasurement};

    double m_horizontalFieldOfView{0U}; //!< Horizontal field of view of the sensor [rad].
    double m_verticalFieldOfView{0U}; //!< Vertical field of view of the sensor [rad].

    Vector3<double> m_mountingPosition{}; //!< Mounting position of the sensor in the VRC [m].
    Vector3<double> m_mountingOrientation{}; //!< Rotations of the sensor mounting pose in the VRC [rad].

    double m_horizontalResolution{0U}; //!< Horizontal angular resolution of the sensor [rad].
    double m_verticalResolution{0U}; //!< Vertical angular resolution of the sensor [rad].

    //========================================
    //!\brief Uncertainty matrix of the mounting position and orientation.
    //----------------------------------------
    Matrix6x6<float> m_mountingPositionOrientationSigma{};

}; // ScannerInfoIn2340

//==============================================================================

//==============================================================================
//! \brief Write string representation for CoordinateSystem value in output stream.
//!
//! \param[in,out] outputStream      The output stream.
//! \param[in]     coordinateSystem  The CoordinateSystem value.
//! \return The same output stream instance as parameter outputStream.
//------------------------------------------------------------------------------
std::ostream& operator<<(std::ostream& outputStream, const ScannerInfoIn2340::CoordinateSystem coordinateSystem);

//==============================================================================
//! \brief Checks for equality.
//!
//! \param[in] lhs  The object, that shall be compared.
//! \param[in] rhs  The object, that first object shall be compared to.
//! \return \c True, if the objects are identical, \c false otherwise.
//------------------------------------------------------------------------------
bool operator==(const ScannerInfoIn2340& lhs, const ScannerInfoIn2340& rhs);

//==============================================================================
//! \brief Checks for inequality.
//!
//! \param[in] lhs  The object, that shall be compared.
//! \param[in] rhs  The object, that first object shall be compared to.
//! \return \c True, if the objects are not identical, \c false otherwise.
//------------------------------------------------------------------------------
inline bool operator!=(const ScannerInfoIn2340& lhs, const ScannerInfoIn2340& rhs) { return !(lhs == rhs); }

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
