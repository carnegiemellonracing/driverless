//==============================================================================
//! \file
//!\verbatim
//! $$MICROVISION_LICENSE_BEGIN$$
//! Copyright (c) 2025 MicroVision, Inc., Redmond, U.S.A.
//! All Rights Reserved.
//!
//! For more details, please refer to the accompanying file
//! License.txt.
//! $$MICROVISION_LICENSE_END$$
//!
//! \date 	11.08.2017
//!
//!\endverbatim
//! VehicleStateContainer
//!\brief Generalized ContainerClass for VehicleState
//------------------------------------------------------------------------------

//==============================================================================

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/DataContainerBase.hpp>
#include <microvision/common/sdk/datablocks/PositionWgs84.hpp>

#include <microvision/common/sdk/NtpTime.hpp>
#include <microvision/common/sdk/Math.hpp>
#include <microvision/common/sdk/Vector3.hpp>
#include <microvision/common/sdk/Matrix3x3.hpp>
#include <microvision/common/sdk/Matrix6x6.hpp>
#include <microvision/common/sdk/PositionUtm.hpp>

#include <microvision/common/sdk/misc/SdkHash.hpp>
#include <microvision/common/sdk/RotationMatrix3d.hpp>
#include <microvision/common/sdk/Quaternion.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! This container holds the current state of the vehicle.
//!
//! Special data types:
//! \ref microvision::common::sdk::VehicleState2805
//! \ref microvision::common::sdk::VehicleState2806
//! \ref microvision::common::sdk::VehicleState2807
//! \ref microvision::common::sdk::VehicleState2808
//------------------------------------------------------------------------------
class VehicleState final : public DataContainerBase
{
    template<class ContainerType, DataTypeId::DataType id>
    friend class Importer;
    template<class ContainerType, DataTypeId::DataType id>
    friend class Exporter;

public:
    constexpr static const char* const containerType{"sdk.generalcontainer.vehiclestate"};

    static constexpr uint64_t getClassIdHashStatic() { return hash(containerType); }

public:
    VehicleState(DataTypeId::DataType importerType = DataTypeId::DataType_Unknown);
    ~VehicleState() override = default;

public:
    uint64_t getClassIdHash() const override { return getClassIdHashStatic(); }

public: // getter
    //========================================
    //!\brief Get the microseconds since startup.
    //!\return microseconds since startup.
    //----------------------------------------
    uint32_t getMicrosecondsSinceStartup() const { return m_microsecondsSinceStartup; }
    //========================================
    //!\brief Get the timestamp of the vehicle state.
    //!\return timestamp of the vehicle state.
    //----------------------------------------
    const NtpTime& getTimestamp() const { return m_timestamp; }
    const NtpTime& getTimestamp(bool& ok) const
    {
        ok = !m_timestamp.is_not_a_date_time();
        return m_timestamp;
    }

    //========================================
    //!\brief Get the sources of the vehicle state.
    //!\return bitmask ssources of the vehicle state.
    //----------------------------------------
    uint16_t getSources() const { return m_sources; }

    //========================================
    //!\brief Get the blind prediction age of the vehicle state
    //!\return blind prediction age of the vehicle state
    //----------------------------------------
    uint16_t getBlindPredictionAge() const { return m_blindPredictionAge; }

    //========================================
    //!\brief Get the validation of the vehicle state.
    //!\return bitmask validation of the vehicle state.
    //----------------------------------------
    uint16_t getValidation() const { return m_validation; }

    //========================================
    //!\brief Get the coordinateSystem type of the vehicle state
    //!\return bitmask coordinateSystem of the vehicle state
    //----------------------------------------
    uint16_t getCoordinateSystem() const { return m_coordinateSystem; }

    //========================================
    //!\brief Get the origin position stored in WGS84 of the vehicle state.
    //!\return originWgs84 of the vehicle state.
    //----------------------------------------
    PositionWgs84 getOriginWgs84() const { return m_originWgs84; }

    //========================================
    //!\brief Get the origin position stored in Utm of the vehicle state.
    //!\return originUtm of the vehicle state.
    //----------------------------------------
    PositionUtm getOriginUtm() const { return m_originUtm; }

    //========================================
    //!\brief Get the origin position stored in an internal Map of the vehicle state.
    //!\return originInternalMap of the vehicle state.
    //----------------------------------------
    std::string getOriginInternalMap() const { return m_originInternalMap; }

    //========================================
    //!\brief Get the actual position of the vehicle state.
    //!\return position of the vehicle state.
    //----------------------------------------
    const Vector3<double>& getRelativePosition() const { return m_relativePosition; }

    //========================================
    //! \brief Get the global vehicle position as a WGS84 coordinate.
    //!
    //! This function only works with VehicleState2809 or newer.
    //! CoordinateSystem flag must be set to either "Utm" or "TangentialPlane".
    //! \param[out] globalCoord  Global vehicle position as WGS84.
    //! \return True, if global coordinate calculation was successful.
    //----------------------------------------
    bool getGlobalPositionAsWgs84(PositionWgs84& globalCoordWgs) const;

    //========================================
    //! \brief Get the global vehicle position as an UTM coordinate.
    //!
    //! This function only works with VehicleState2809 or newer.
    //! CoordinateSystem flag must be set to either "Utm" or "TangentialPlane".
    //! \param[out] globalCoord  Global vehicle position as UTM.
    //! \return True, if global coordinate calculation was successful.
    //----------------------------------------
    bool getGlobalPositionAsUtm(PositionUtm& globalCoordUtm) const;

    //========================================
    //!\brief Get the sigma of the position and orientation of the vehicle state.
    //!\return positionorientationsigma of the vehicle state.
    //----------------------------------------
    const Matrix6x6<double>& getPositionOrientationSigma() const { return m_positionOrientationSigma; }

    //========================================
    //!\brief Get the curse angle of the vehicle state in rad.
    //!\return course angle of the vehicle state.
    //----------------------------------------
    float getCourseAngle() const { return m_courseAngle; }
    float getCourseAngle(bool& ok) const
    {
        ok = !std::isnan(m_courseAngle);
        return m_courseAngle;
    }

    //========================================
    //!\brief Get the sigma of the curse angle of the vehicle state in rad.
    //!\return course angle sigma of the vehicle state.
    //----------------------------------------
    float getCourseAngleSigma() const { return m_courseAngleSigma; }
    float getCourseAngleSigma(bool& ok) const
    {
        ok = !std::isnan(m_courseAngleSigma);
        return m_courseAngleSigma;
    }
    float getHeadingAngle() const { return m_headingAngle; }
    float getHeadingAngle(bool& ok) const
    {
        ok = !std::isnan(m_headingAngle);
        return m_headingAngle;
    }
    float getHeadingAngleSigma() const { return m_headingAngleSigma; }
    float getHeadingAngleSigma(bool& ok) const
    {
        ok = !std::isnan(m_headingAngleSigma);
        return m_headingAngleSigma;
    }

    //========================================
    //!\brief Get the actual orientation of the vehicle state as euler vector.
    //!\return orientation of the vehicle state.
    //----------------------------------------
    const Vector3<double> getOrientationAsVector() const
    {
        return m_orientationAsRotationMatrix.getEulerAnglesWithRotationOrderRollPitchYaw();
    }

    //========================================
    //!\brief Get the actual orientation of the vehicle state as rotation matrix.
    //!\return orientation of the vehicle state.
    //----------------------------------------
    const RotationMatrix3d<double>& getOrientationAsRotationMatrix() const { return m_orientationAsRotationMatrix; }

    //========================================
    //!\brief Get the actual orientation of the vehicle state as quaternion.
    //!\return orientation of the vehicle state.
    //----------------------------------------
    const Quaternion<double> getOrientationAsQuaternion() const
    {
        return m_orientationAsRotationMatrix.getQuaternion();
    }

    //========================================
    //!\brief Get the linear velocity of the vehicle state in m/s.
    //!\return velocity  of the vehicle state.
    //----------------------------------------
    const Vector3<float>& getLinearVelocity() const { return m_linearVelocity; }

    //========================================
    //!\brief Get the sigma of the linear velocity of the vehicle state in  m/s.
    //!\return velocity sigma of the vehicle state.
    //----------------------------------------
    const Matrix3x3<float>& getLinearVelocitySigma() const { return m_linearVelocitySigma; }

    //========================================
    //!\brief Get the angular velocity of the vehicle state in rad/s.
    //!\return velocity  of the vehicle state.
    //----------------------------------------
    const Vector3<float>& getAngularVelocity() const { return m_angularVelocity; }

    //========================================
    //!\brief Get the sigma of the angular velocity of the vehicle state in  rad/s
    //!\return velocity sigma of the vehicle state
    //----------------------------------------
    const Matrix3x3<float>& getAngularVelocitySigma() const { return m_angularVelocitySigma; }

    //========================================
    //!\brief Get the linear acceleration of the vehicle state in  m/s^2.
    //!\return acceleration of the vehicle state.
    //----------------------------------------
    const Vector3<float>& getAcceleration() const { return m_acceleration; }

    //========================================
    //!\brief Get the sigma of the linear acceleration of the vehicle state in  m/s^2.
    //!\return acceleration sigma of the vehicle state.
    //----------------------------------------
    const Matrix3x3<float>& getAccelerationSigma() const { return m_accelerationSigma; }

    //========================================
    //!\brief Get the vehicle body to road surface position of the vehicle state in m.
    //!\return vehicle body to road surface position of the vehicle state.
    //----------------------------------------
    const Vector3<double>& getVehicleBodyToRoadSurfacePosition() const { return m_vehicleBodyToRoadSurfacePosition; }

    //========================================
    //!\brief Get the vehicle body to road surface orientation of the vehicle state in rad as euler vector.
    //!\return vehicle body to road surface orientation of the vehicle state.
    //----------------------------------------
    const Vector3<double> getVehicleBodyToRoadSurfaceOrientationAsVector() const
    {
        return m_vehicleBodyToRoadSurfaceOrientationAsRotationMatrix.getEulerAnglesWithRotationOrderRollPitchYaw();
    }

    //========================================
    //!\brief Get the sigma of the vehicle body to road surface position and orientation of the vehicle state.
    //!\return sigma of the vehicle body to road surface position and orientation of the vehicle state.
    //----------------------------------------
    const Matrix6x6<double>& getVehicleBodyToRoadSurfacePositionOrientationSigma() const
    {
        return m_vehicleBodyToRoadSurfacePositionOrientationSigma;
    }

    //========================================
    //!\brief Get the vehicle body to road surface orientation of the vehicle state as rotation matrix.
    //!\return vehicle body to road surface orientation of the vehicle state.
    //----------------------------------------
    const RotationMatrix3d<double>& getVehicleBodyToRoadSurfaceOrientationAsRotationMatrix() const
    {
        return m_vehicleBodyToRoadSurfaceOrientationAsRotationMatrix;
    }

    //========================================
    //!\brief Get the vehicle body to road surface orientation of the vehicle state as quaternion.
    //!\return vehicle body to road surface orientation of the vehicle state.
    //----------------------------------------
    const Quaternion<double> getVehicleBodyToRoadSurfaceOrientationAsQuaternion() const
    {
        return m_vehicleBodyToRoadSurfaceOrientationAsRotationMatrix.getQuaternion();
    }

    //========================================
    //!\brief Get the steering angle of the vehicle state in rad.
    //!\return steering angle of the vehicle state.
    //----------------------------------------
    float getSteerAngle() const { return m_steerAngle; }
    float getSteerAngle(bool& ok) const
    {
        ok = !std::isnan(m_steerAngle);
        return m_steerAngle;
    }

    //========================================
    //!\brief Get the steering wheel angle of the vehicle state in rad.
    //!\return steering wheel angle of the vehicle state.
    //----------------------------------------
    float getSteeringWheelAngle() const { return m_steeringWheelAngle; }
    float getSteeringWheelAngle(bool& ok) const
    {
        ok = !std::isnan(m_steeringWheelAngle);
        return m_steeringWheelAngle;
    }

    //========================================
    //!\brief Get the vehicle width of the vehicle state in m.
    //!\return vehicle width of the vehicle state.
    //----------------------------------------
    float getVehicleWidth() const { return m_vehicleWidth; }
    float getVehicleWidth(bool& ok) const
    {
        ok = !std::isnan(m_vehicleWidth);
        return m_vehicleWidth;
    }

    //========================================
    //!\brief Get the minTurningCircle of the vehicle state in m.
    //!\return minTurningCircle of the vehicle state.
    //----------------------------------------
    float getMinTurningCircle() const { return m_minTurningCircle; }
    float getMinTurningCircle(bool& ok) const
    {
        ok = !std::isnan(m_minTurningCircle);
        return m_minTurningCircle;
    }

    //========================================
    //!\brief Get the vehicleFrontToFrontAxle of the vehicle state in m.
    //!\return vehicleFrontToFrontAxle of the vehicle state.
    //----------------------------------------
    float getVehicleFrontToFrontAxle() const { return m_vehicleFrontToFrontAxle; }
    float getVehicleFrontToFrontAxle(bool& ok) const
    {
        ok = !std::isnan(m_vehicleFrontToFrontAxle);
        return m_vehicleFrontToFrontAxle;
    }

    //========================================
    //!\brief Get the frontAxleToRearAxle of the vehicle state in m.
    //!\return frontAxleToRearAxle of the vehicle state.
    //----------------------------------------
    float getFrontAxleToRearAxle() const { return m_frontAxleToRearAxle; }
    float getFrontAxleToRearAxle(bool& ok) const
    {
        ok = !std::isnan(m_frontAxleToRearAxle);
        return m_frontAxleToRearAxle;
    }

    //========================================
    //!\brief Get the rearAxleToVehicleRear of the vehicle state in m.
    //!\return rearAxleToVehicleRear of the vehicle state.
    //----------------------------------------
    float getRearAxleToVehicleRear() const { return m_rearAxleToVehicleRear; }
    float getRearAxleToVehicleRear(bool& ok) const
    {
        ok = !std::isnan(m_rearAxleToVehicleRear);
        return m_rearAxleToVehicleRear;
    }

    //========================================
    //!\brief Get the rearAxleToOrigin of the vehicle state in m.
    //!\return rearAxleToOrigin of the vehicle state.
    //----------------------------------------
    float getRearAxleToOrigin() const { return m_rearAxleToOrigin; }
    float getRearAxleToOrigin(bool& ok) const
    {
        ok = !std::isnan(m_rearAxleToOrigin);
        return m_rearAxleToOrigin;
    }

    //========================================
    //!\brief Get the total distance driven by the vehicle.
    //!\return distance of the vehicle state.
    //----------------------------------------
    double getTotalDistanceDriven() const { return m_totalDistanceDriven; }
    double getTotalDistanceDriven(bool& ok) const
    {
        ok = !std::isnan(m_totalDistanceDriven);
        return m_totalDistanceDriven;
    }

    //========================================
    //!\brief Get the steer ratio coefficient 0 of the vehicle state.
    //!\return steer ratio coefficient 0 of the vehicle state.
    //----------------------------------------
    float getSteerRatioCoeff0() const { return m_steerRatioCoeff0; }
    float getSteerRatioCoeff0(bool& ok) const
    {
        ok = !std::isnan(m_steerRatioCoeff0);
        return m_steerRatioCoeff0;
    }

    //========================================
    //!\brief Get the steer ratio coefficient 1 of the vehicle state.
    //!\return steer ratio coefficient 1 of the vehicle state.
    //----------------------------------------
    float getSteerRatioCoeff1() const { return m_steerRatioCoeff1; }
    float getSteerRatioCoeff1(bool& ok) const
    {
        ok = !std::isnan(m_steerRatioCoeff1);
        return m_steerRatioCoeff1;
    }

    //========================================
    //!\brief Get the steer ratio coefficient 2 of the vehicle state.
    //!\return steer ratio coefficient 2 of the vehicle state.
    //----------------------------------------
    float getSteerRatioCoeff2() const { return m_steerRatioCoeff2; }
    float getSteerRatioCoeff2(bool& ok) const
    {
        ok = !std::isnan(m_steerRatioCoeff2);
        return m_steerRatioCoeff2;
    }

    //========================================
    //!\brief Get the steer ratio coefficient 3 of the vehicle state.
    //!\return steer ratio coefficient 3 of the vehicle state.
    //----------------------------------------
    float getSteerRatioCoeff3() const { return m_steerRatioCoeff3; }
    float getSteerRatioCoeff3(bool& ok) const
    {
        ok = !std::isnan(m_steerRatioCoeff3);
        return m_steerRatioCoeff3;
    }

public: // setter
    //========================================
    //!\brief Set the microseconds since startup.
    //!\param[in] microseconds  New value for microseconds since startup.
    //----------------------------------------
    void setMicroseconds(const uint32_t microseconds) { m_microsecondsSinceStartup = microseconds; }

    //========================================
    //!\brief Set the timestamp of the vehicle state.
    //!\param[in] timestamp  New timestamp of the vehicle state.
    //----------------------------------------
    void setTimestamp(const NtpTime& timestamp) { m_timestamp = timestamp; }

    //========================================
    //!\brief Set the sources of the vehicle state.
    //!\param[in] sources  New sources of the vehicle state (bitmask).
    //----------------------------------------
    void setSources(const uint16_t sources) { m_sources = sources; }

    //========================================
    //!\brief Set the blind prediction age of the vehicle state.
    //!\param[in] blindPredictionAge  New blind prediction age of the vehicle state.
    //----------------------------------------
    void setBlindPredictionAge(const uint16_t blindPredictionAge) { m_blindPredictionAge = blindPredictionAge; }

    //========================================
    //!\brief Set the validation of the vehicle state.
    //!\param[in] validation  New validation of the vehicle state (bitmask).
    //----------------------------------------
    void setValidation(const uint16_t validation) { m_validation = validation; }

    //========================================
    //!\brief Set the coordinateSystem type of the vehicle state.
    //!\param[in] coordinateSystem  New coordinateSystem type of the vehicle state (bitmask).
    //----------------------------------------
    void setCoordinateSystem(const uint16_t coordinateSystem) { m_coordinateSystem = coordinateSystem; }

    //========================================
    //!\brief Set the origin position stored in WGS84 of the vehicle state.
    //!\param[in] origin New origin position of the vehicle state.
    //----------------------------------------
    void setOriginWgs84(const PositionWgs84& origin) { m_originWgs84 = origin; }

    //========================================
    //!\brief Set the origin position stored in Utm of the vehicle state.
    //!\param[in] origin New origin position of the vehicle state.
    //----------------------------------------
    void setOriginUtm(const PositionUtm& origin) { m_originUtm = origin; }

    //========================================
    //!\brief Set the origin position stored in an internal map of the vehicle state.
    //!\param[in] origin New origin position of the vehicle state.
    //----------------------------------------
    void setOriginInternalMap(const std::string& origin) { m_originInternalMap = origin; }

    //========================================
    //!\brief Set the actual position  of the vehicle state.
    //!\param[in] position New position of the vehicle state.
    //----------------------------------------
    void setRelativePosition(const Vector3<double>& position) { m_relativePosition = position; }

    //========================================
    //!\brief Set the actual position and orientation as euler vector sigma of the vehicle state.
    //!\param[in] sigma New sigma of the vehicle state.
    //----------------------------------------
    void setPositionOrientationSigma(const Matrix6x6<double>& sigma) { m_positionOrientationSigma = sigma; }

    //========================================
    //!\brief Set the course angle of the vehicle state.
    //!\param[in] courseAngle New course angle of the vehicle state.
    //----------------------------------------
    void setCourseAngle(const float courseAngle) { m_courseAngle = courseAngle; }

    //========================================
    //!\brief Set the sigma of the course angle of the vehicle state.
    //!\param[in] courseAngleSigma New course angle sigma of the vehicle state.
    //----------------------------------------
    void setCourseAngleSigma(const float courseAngleSigma) { m_courseAngleSigma = courseAngleSigma; }

    void setHeadingAngle(const float headingAngle) { m_headingAngle = headingAngle; }
    void setHeadingAngleSigma(const float headingAngleSigma) { m_headingAngleSigma = headingAngleSigma; }

    //========================================
    //!\brief Set the orientation of the vehicle state as euler vector.
    //!\param[in] orientation New orientation of the vehicle state.
    //----------------------------------------
    void setOrientationAsVector(const Vector3<double>& orientation)
    {
        m_orientationAsRotationMatrix.setFromVectorWithRotationOrderRollPitchYaw(orientation);
    }

    //========================================
    //!\brief Set the orientation of the vehicle state as rotation matrix.
    //!\param[in] orientation New orientation of the vehicle state.
    //----------------------------------------
    void setOrientationAsRotationMatrix(const RotationMatrix3d<double>& orientation)
    {
        m_orientationAsRotationMatrix = orientation;
    }

    //========================================
    //!\brief Set the orientation of the vehicle state as quaternion.
    //!\param[in] orientation New orientation of the vehicle state.
    //----------------------------------------
    void setOrientationAsQuaternion(const Quaternion<double>& orientation)
    {
        m_orientationAsRotationMatrix = orientation.getRotationMatrix();
    }

    //========================================
    //!\brief Set the linear velocity of the vehicle state in m/s.
    //!\param[in] velocity New velocity  of the vehicle state.
    //----------------------------------------
    void setLinearVelocity(const Vector3<float>& velocity) { m_linearVelocity = velocity; }

    //========================================
    //!\brief Set the sigma of the linear velocity of the vehicle state in  m/s.
    //!\param[in] sigma New velocity sigma of the vehicle state.
    //----------------------------------------
    void setLinearVelocitySigma(const Matrix3x3<float>& sigma) { m_linearVelocitySigma = sigma; }

    //========================================
    //!\brief Set the angular velocity of the vehicle state in rad/s.
    //!\param[in] velocity New velocity  of the vehicle state.
    //----------------------------------------
    void setAngularVelocity(const Vector3<float>& velocity) { m_angularVelocity = velocity; }

    //========================================
    //!\brief Set the sigma of the angular velocity of the vehicle state in  rad/s.
    //!\param[in] sigma  New velocity sigma of the vehicle state.
    //----------------------------------------
    void setAngularVelocitySigma(const Matrix3x3<float>& sigma) { m_angularVelocitySigma = sigma; }

    //========================================
    //!\brief Set the linear acceleration of the vehicle state in  m/s^2.
    //!\param[in] acceleration New acceleration of the vehicle state.
    //----------------------------------------
    void setAcceleration(const Vector3<float>& acceleration) { m_acceleration = acceleration; }

    //========================================
    //!\brief Set the sigma of the linear acceleration of the vehicle state in  m/s^2.
    //!\param[in] sigma New acceleration sigma of the vehicle state.
    //----------------------------------------
    void setAccelerationSigma(const Matrix3x3<float>& sigma) { m_accelerationSigma = sigma; }

    //========================================
    //!\brief Set the vehicle body to road surface position of the vehicle state in m.
    //!\param[in] position  New vehicle body to road surface position of the vehicle state.
    //----------------------------------------
    void setVehicleBodyToRoadSurfacePosition(const Vector3<double>& position)
    {
        m_vehicleBodyToRoadSurfacePosition = position;
    }

    //========================================
    //!\brief Set the vehicle body to road surface orientation of the vehicle state in rad as euler vector.
    //!\param[in] orientation New vehicle body to road surface orientation of the vehicle state.
    //----------------------------------------
    void setVehicleBodyToRoadSurfaceOrientationAsVector(const Vector3<double>& orientation)
    {
        m_vehicleBodyToRoadSurfaceOrientationAsRotationMatrix.setFromVectorWithRotationOrderRollPitchYaw(orientation);
    }

    //========================================
    //!\brief Set the sigma of the vehicle body to road surface position and orientation of the vehicle state.
    //!\param[in]  sigma New sigma of the vehicle body to road surface position and orientation of the vehicle state.
    //----------------------------------------
    void setVehicleBodyToRoadSurfacePositionOrientationSigma(const Matrix6x6<double>& sigma)
    {
        m_vehicleBodyToRoadSurfacePositionOrientationSigma = sigma;
    }

    //========================================
    //!\brief Set the vehicle body to road surface orientation of the vehicle state as rotation matrix.
    //!\param[in]  orientation New vehicle body to road surface orientation of the vehicle state.
    //----------------------------------------
    void setVehicleBodyToRoadSurfaceOrientationAsRotationMatrix(const RotationMatrix3d<double>& orientation)
    {
        m_vehicleBodyToRoadSurfaceOrientationAsRotationMatrix = orientation;
    }

    //========================================
    //!\brief Set the vehicle body to road surface orientation of the vehicle state as quaternion
    //!\param[in] orientation New vehicle body to road surface orientation of the vehicle state
    //----------------------------------------
    void setVehicleBodyToRoadSurfaceOrientationAsQuaternion(const Quaternion<double>& orientation)
    {
        m_vehicleBodyToRoadSurfaceOrientationAsRotationMatrix = orientation.getRotationMatrix();
    }

    //========================================
    //!\brief Set the distance by the vehicle.
    //!\param[in] dsitans New distance.
    //----------------------------------------
    void setTotalDistanceDriven(const double distance) { m_totalDistanceDriven = distance; }

    //========================================
    //!\brief Set the steering wheel angle of the vehicle state in rad.
    //!\param[in] steeringWheelAngle New steeringWheelAngle of the vehicle state.
    //----------------------------------------
    void setSteeringWheelAngle(const float steeringWheelAngle) { m_steeringWheelAngle = steeringWheelAngle; }

    //========================================
    //!\brief Set the steer angle of the vehicle state in rad.
    //!\param[in] steerAngle New steer angle of the vehicle state.
    //----------------------------------------
    void setSteerAngle(const float steerAngle) { m_steerAngle = steerAngle; }

    //========================================
    //!\brief Set the vehicle width of the vehicle state in m.
    //!\param[in] vehicleWidth New vehicle width of the vehicle state.
    //----------------------------------------
    void setVehicleWidth(const float vehicleWidth) { m_vehicleWidth = vehicleWidth; }

    //========================================
    //!\brief Set the min turning circle of the vehicle state in m.
    //!\param[in]  minTurningCircle New min turning circle of the vehicle state.
    //----------------------------------------
    void setMinTurningCircle(const float minTurningCircle) { m_minTurningCircle = minTurningCircle; }

    //========================================
    //!\brief Set the distance from the vehicle front to front axle of the vehicle state in m.
    //!\param[in] vehicleFrontToFrontAxle New distance.
    //----------------------------------------
    void setVehicleFrontToFrontAxle(const float vehicleFrontToFrontAxle)
    {
        m_vehicleFrontToFrontAxle = vehicleFrontToFrontAxle;
    }

    //========================================
    //!\brief Set the distance from the front axle to rear axle of the vehicle state in m.
    //!\param[in] rontAxleToRearAxle New distance.
    //----------------------------------------
    void setFrontAxleToRearAxle(const float frontAxleToRearAxle) { m_frontAxleToRearAxle = frontAxleToRearAxle; }

    //========================================
    //!\brief Set the distance from the rear axle to vehicle rear of the vehicle state in m.
    //!\param[in] rearAxleToVehicleRear New distance.
    //----------------------------------------
    void setRearAxleToVehicleRear(const float rearAxleToVehicleRear)
    {
        m_rearAxleToVehicleRear = rearAxleToVehicleRear;
    }

    //========================================
    //!\brief Set the distance from the rear axle to origin of the vehicle state in m.
    //!\param[in] rearAxleToOrigin New distance.
    //----------------------------------------
    void setRearAxleToOrigin(const float rearAxleToOrigin) { m_rearAxleToOrigin = rearAxleToOrigin; }

    //========================================
    //!\brief Set the steer ratio coefficient 0 of the vehicle state.
    //!\param[in] steerRatioCoeff0 New coefficient of the vehicle state.
    //----------------------------------------
    void setSteerRatioCoeff0(const float steerRatioCoeff0) { m_steerRatioCoeff0 = steerRatioCoeff0; }

    //========================================
    //!\brief Set the steer ration coefficient 1 of the vehicle state.
    //!\param[in] steerRatioCoeff1 New coefficient of the vehicle state.
    //----------------------------------------
    void setSteerRatioCoeff1(const float steerRatioCoeff1) { m_steerRatioCoeff1 = steerRatioCoeff1; }

    //========================================
    //!\brief Set the steer ration coefficient 2 of the vehicle state.
    //!\param[in] steerRatioCoeff2 New coefficient of the vehicle state.
    //----------------------------------------
    void setSteerRatioCoeff2(const float steerRatioCoeff2) { m_steerRatioCoeff2 = steerRatioCoeff2; }

    //========================================
    //!\brief Set the steer ration coefficient 3 of the vehicle state.
    //!\param[in] steerRatioCoeff3 New coefficient of the vehicle state.
    //----------------------------------------
    void setSteerRatioCoeff3(const float steerRatioCoeff3) { m_steerRatioCoeff3 = steerRatioCoeff3; }

protected: //members
    uint32_t m_microsecondsSinceStartup{0}; //!< microseconds since startup.
    NtpTime m_timestamp{0}; //!< timestamp of this data
    uint16_t m_sources{0};
    uint16_t m_blindPredictionAge{0};
    uint16_t m_validation{0}; //!< The validation status (Bitmask)
    uint16_t m_coordinateSystem{0}; //!< coordinate system for this frame (Bitmask)

    PositionWgs84 m_originWgs84; //!< reference point for GlobalTangentailPlane
    PositionUtm m_originUtm; //!< Utm coordinates
    std::string m_originInternalMap; //!< Name of map containing map coordinates.

    Vector3<double> m_relativePosition; //!< Absolute Position from origin [m] (x,y,z)
    RotationMatrix3d<double> m_orientationAsRotationMatrix; //!< current vehicle angle
    Matrix6x6<double> m_positionOrientationSigma; //!< standard deviation [m²/rad²] (x,y,z,roll,pitch,yaw)

    float m_courseAngle{NaN}; //!< Absolute orientation at time timeStamp | [rad]
    float m_courseAngleSigma{NaN}; //!< [rad]
    float m_headingAngle{NaN}; //!< heading angle [rad] - is different to course angle
    float m_headingAngleSigma{NaN}; //!< standard deviation [rad]

    Vector3<float> m_linearVelocity; //!< Current velocity of the vehicle [m/s] (x,y,z)
    Matrix3x3<float> m_linearVelocitySigma; //!< standard deviation [m/s] (x,y,z)
    Vector3<float> m_angularVelocity; //!< Current angular velocity of the vehicle [rad/s]   (roll,pitch,yaw)
    Matrix3x3<float> m_angularVelocitySigma; //!< standard deviation [rad/s]  (roll,pitch,yaw)

    Vector3<float> m_acceleration; //!< Current acceleration of the vehicle [m/s^2] (x,y,z)
    Matrix3x3<float> m_accelerationSigma; //!< standard deviation [m/s^2] (x,y,z)

    Vector3<double> m_vehicleBodyToRoadSurfacePosition; //!< Distance of vehicle body to the road surface [m] (x,y,z)
    Matrix6x6<double>
        m_vehicleBodyToRoadSurfacePositionOrientationSigma; //!< standard deviation [???] (x,y,z,roll,pitch,yaw)
    RotationMatrix3d<double>
        m_vehicleBodyToRoadSurfaceOrientationAsRotationMatrix; //!< Orientation of vehicle body to the road surface

    float m_steerAngle{NaN}; //!< [rad]
    float m_steeringWheelAngle{NaN}; //!< steering wheel angle [rad]

    float m_vehicleWidth{NaN}; //!< Vehicle width [m]
    float m_minTurningCircle{NaN};
    double m_totalDistanceDriven{NaN_double}; //!< Distance: total distance driven [m]

    float m_vehicleFrontToFrontAxle{NaN}; //!< Distance: vehicle's front axle to vehicle's front [m]
    float m_frontAxleToRearAxle{NaN}; //!< Distance: vehicle's rear axle to vehicle's front axle [m]
    float m_rearAxleToVehicleRear{NaN}; //!< Distance: vehicle's rear axle to vehicle's rear [m]
    float m_rearAxleToOrigin{NaN}; //!< Distance: vehicle's rear axle to origin [m]
    float m_steerRatioCoeff0{NaN}; //!< s_3*x^3 + s_2*x^2 + s-2*x^1 + s_0
    float m_steerRatioCoeff1{NaN}; //!< s_3*x^3 + s_2*x^2 + s-2*x^1 + s_0
    float m_steerRatioCoeff2{NaN}; //!< s_3*x^3 + s_2*x^2 + s-2*x^1 + s_0
    float m_steerRatioCoeff3{NaN}; //!< s_3*x^3 + s_2*x^2 + s-2*x^1 + s_0
}; // VehicleState

//==============================================================================

bool operator==(const VehicleState& lhs, const VehicleState& rhs);
bool operator!=(const VehicleState& lhs, const VehicleState& rhs);

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
