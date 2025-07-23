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
#include <microvision/common/sdk/misc/SdkHash.hpp>

#include <microvision/common/sdk/datablocks/DataContainerBase.hpp>
#include <microvision/common/sdk/datablocks/PositionWgs84.hpp>

#include <microvision/common/sdk/PositionUtm.hpp>
#include <microvision/common/sdk/Math.hpp>
#include <microvision/common/sdk/Matrix3x3.hpp>
#include <microvision/common/sdk/Matrix6x6.hpp>
#include <microvision/common/sdk/RotationMatrix3d.hpp>
#include <microvision/common/sdk/Vector3.hpp>
#include <microvision/common/sdk/Quaternion.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Vehicle state
//!
//! extended Vehicle state 2808 data available direct from system or from lane algorithms.
//!
//! General data type: \ref microvision::common::sdk::VehicleState
//------------------------------------------------------------------------------
class VehicleState2809 final : public SpecializedDataContainer
{
    template<class ContainerType, DataTypeId::DataType id>
    friend class Importer;
    template<class ContainerType, DataTypeId::DataType id>
    friend class Exporter;

public:
    //========================================
    //! \brief Container type as string.
    //----------------------------------------
    constexpr static const char* const containerType{"sdk.specialcontainer.vehiclestate2809"};

    //========================================
    //! \brief Hash value of this container (static version).
    //----------------------------------------
    static constexpr uint64_t getClassIdHashStatic() { return hash(containerType); }

    //========================================
    //! \brief Hash value of this container.
    //----------------------------------------
    uint64_t getClassIdHash() const override { return getClassIdHashStatic(); }

public:
    //========================================
    //! \brief Constructor.
    //----------------------------------------
    VehicleState2809() = default;

    //========================================
    //! \brief Destructor.
    //----------------------------------------
    ~VehicleState2809() override = default;

public:
    //========================================
    //!\brief flags to be used to signal VehicleState source
    //----------------------------------------
    enum class VehicleStateSource : uint16_t
    {
        Unknown              = 0x0000U, //!< unknown source
        Can                  = 0x0001U, //!< from CAN data
        Gps                  = 0x0002U, //!< GPS based
        Imu                  = 0x0004U, //!< calculated from IMU
        LidarOdometry        = 0x0008U, //!< based on lidar odometry
        LandmarkLocalization = 0x0010U, //!< landmark based estimation
        VisualOdometry       = 0x0020U, //!< visual odometry based
        ReferenceEgostate    = 0x0040U, //!< calculated with reference ego-state
        OnlineFilter         = 0x0080U, //!< online filter based estimate
        Imported             = 0x0100U //!< imported data, not self-estimated
    };

    //========================================
    //!\brief flags to be used to signal vehicle state quality
    //----------------------------------------
    enum class VehicleStateValidation : uint16_t
    {
        Ok                            = 0x0000U,
        ConsistencyCheckFailed        = 0x0001U,
        PoseEstimateUnderconstraint   = 0x0002U,
        MotionEstimateUnderconstraint = 0x0004U
    };

    //========================================
    //!\brief flags to be used to signal which coordinate system is used
    //----------------------------------------
    enum class VehicleStateCoordinateSystem : uint16_t
    {
        Unknown               = 0x0000U,
        GlobalTangentialPlane = 0x0001U,
        Utm                   = 0x0002U,
        RelativeCoordinates   = 0x0004U,
        MapCoordinates        = 0x0008U
    };

public: // getter
    //========================================
    //!\brief Get the microseconds since startup
    //!\return microseconds since startup
    //----------------------------------------
    uint32_t getMicrosecondsSinceStartup() const { return m_microsecondsSinceStartup; }

    //========================================
    //!\brief Get the timestamp of the vehicle state
    //!\return timestamp of the vehicle state
    //----------------------------------------
    NtpTime getTimestamp() const { return m_timestamp; }

    //========================================
    //!\brief Get the sources of the vehicles tate
    //!\return bitmask sources of the vehicle state
    //----------------------------------------
    VehicleStateSource getSources() const { return m_sources; }

    //========================================
    //!\brief Get the blind prediction age of the vehicle state
    //!\return blind prediction age of the vehicle state
    //----------------------------------------
    uint16_t getBlindPredictionAge() const { return m_blindPredictionAge; }

    //========================================
    //!\brief Get the validation of the vehicle state
    //!\return bitmask validation of the vehicle state
    //----------------------------------------
    VehicleStateValidation getValidation() const { return m_validation; }

    //========================================
    //!\brief Get the coordinateSystem type of the vehicle state
    //!\return bitmask coordinateSystem of the vehicle state
    //----------------------------------------
    VehicleStateCoordinateSystem getCoordinateSystem() const { return m_coordinateSystem; }

    //========================================
    //!\brief Get the origin position stored in WGS84 of the vehicle state
    //!\return originWgs84 of the vehicle state
    //----------------------------------------
    PositionWgs84 getOriginWgs84() const { return m_originWgs84; }

    //========================================
    //!\brief Get the origin position stored in Utm of the vehicle state
    //!\return originUtm of the vehicle state
    //----------------------------------------
    PositionUtm getOriginUtm() const { return m_originUtm; }

    //========================================
    //!\brief Get the origin position stored in an internal Map of the vehicle state
    //!\return originInternalMap of the vehicle state
    //----------------------------------------
    std::string getOriginInternalMap() const { return m_originInternalMap; }

    //========================================
    //!\brief Get the actual position of the vehicle state
    //!\return position of the vehicle state
    //----------------------------------------
    const Vector3<double>& getPosition() const { return m_position; }

    //========================================
    //!\brief Get the actual orientation of the vehicle state as Euler vector with the rotation order yaw, pitch, roll
    //!\return Orientation of the vehicle state
    //----------------------------------------
    const Vector3<double> getOrientationAsVector() const
    {
        return m_orientationAsRotationMatrix.getEulerAnglesWithRotationOrderRollPitchYaw();
    }

    //========================================
    //!\brief Get the actual orientation of the vehicle state as rotation matrix
    //!\return Orientation of the vehicle state
    //----------------------------------------
    const RotationMatrix3d<double>& getOrientationAsRotationMatrix() const { return m_orientationAsRotationMatrix; }

    //========================================
    //!\brief Get the actual orientation of the vehicle state as quaternion
    //!\return Orientation of the vehicle state
    //----------------------------------------
    const Quaternion<double> getOrientationAsQuaternion() const
    {
        return m_orientationAsRotationMatrix.getQuaternion();
    }

    //========================================
    //!\brief Get the sigma of the position and orientation of the vehicle state
    //!\return Position orientation-sigma of the vehicle state
    //----------------------------------------
    const Matrix6x6<double>& getPositionOrientationSigma() const { return m_positionOrientationSigma; }

    //========================================
    //!\brief Get the curse angle of the vehicle state in rad
    //!\return Course angle of the vehicle state
    //----------------------------------------
    float getCourseAngle() const { return m_courseAngle; }

    //========================================
    //!\brief Get the sigma of the curse angle of the vehicle state in rad
    //!\return Course angle sigma of the vehicle state
    //----------------------------------------
    float getCourseAngleSigma() const { return m_courseAngleSigma; }

    //========================================
    //!\brief Get the linear velocity of the vehicle state in m/s
    //!\return Velocity  of the vehicle state
    //----------------------------------------
    const Vector3<float>& getLinearVelocity() const { return m_linearVelocity; }

    //========================================
    //!\brief Get the sigma of the linear velocity of the vehicle state in  m/s
    //!\return Velocity sigma of the vehicle state
    //----------------------------------------
    const Matrix3x3<float>& getLinearVelocitySigma() const { return m_linearVelocitySigma; }

    //========================================
    //!\brief Get the angular velocity of the vehicle state in rad/s
    //!\return Velocity  of the vehicle state
    //----------------------------------------
    const Vector3<float>& getAngularVelocity() const { return m_angularVelocity; }

    //========================================
    //!\brief Get the sigma of the angular velocity of the vehicle state in  rad/s
    //!\return Velocity sigma of the vehicle state
    //----------------------------------------
    const Matrix3x3<float>& getAngularVelocitySigma() const { return m_angularVelocitySigma; }

    //========================================
    //!\brief Get the linear acceleration of the vehicle state in  m/s^2
    //!\return Acceleration of the vehicle state
    //----------------------------------------
    const Vector3<float>& getAcceleration() const { return m_acceleration; }

    //========================================
    //!\brief Get the sigma of the linear acceleration of the vehicle state in  m/s^2
    //!\return Acceleration sigma of the vehicle state
    //----------------------------------------
    const Matrix3x3<float>& getAccelerationSigma() const { return m_accelerationSigma; }

    //========================================
    //!\brief Get the vehicle body to road surface position of the vehicle state in  m
    //!\return Vehicle body to road surface position of the vehicle state
    //----------------------------------------
    const Vector3<double>& getVehicleBodyToRoadSurfacePosition() const { return m_vehicleBodyToRoadSurfacePosition; }

    //========================================
    //!\brief Get the vehicle body to road surface orientation of the vehicle state in rad as Euler vector with the rotation order yaw, pitch,roll
    //!\return Vehicle body to road surface orientation of the vehicle state
    //----------------------------------------
    const Vector3<double> getVehicleBodyToRoadSurfaceOrientationAsVector() const
    {
        return m_vehicleBodyToRoadSurfaceOrientationAsRotationMatrix.getEulerAnglesWithRotationOrderRollPitchYaw();
    };

    //========================================
    //!\brief Get the vehicle body to road surface orientation of the vehicle state as rotation matrix
    //!\return Vehicle body to road surface orientation of the vehicle state
    //----------------------------------------
    const RotationMatrix3d<double>& getVehicleBodyToRoadSurfaceOrientationAsRotationMatrix() const
    {
        return m_vehicleBodyToRoadSurfaceOrientationAsRotationMatrix;
    }

    //========================================
    //!\brief Get the vehicle body to road surface orientation of the vehicle state as quaternion
    //!\return Vehicle body to road surface orientation of the vehicle state
    //----------------------------------------
    const Quaternion<double> getVehicleBodyToRoadSurfaceOrientationAsQuaternion() const
    {
        return m_vehicleBodyToRoadSurfaceOrientationAsRotationMatrix.getQuaternion();
    }

    //========================================
    //!\brief Get the sigma of the vehicle body to road surface position and orientation of the vehicle state
    //!\return Sigma of the vehicle body to road surface position and orientation of the vehicle state
    //----------------------------------------
    const Matrix6x6<double>& getVehicleBodyToRoadSurfacePositionOrientationSigma() const
    {
        return m_vehicleBodyToRoadSurfacePositionOrientationSigma;
    }

    //========================================
    //!\brief Get the vehicle width of the vehicle state in m
    //!\return Vehicle width of the vehicle state
    //----------------------------------------
    float getVehicleWidth() const { return m_vehicleWidth; }

    //========================================
    //!\brief Get the minTurningCircle of the vehicle state in m
    //!\return minTurningCircle of the vehicle state
    //----------------------------------------
    float getMinTurningCircle() const { return m_minTurningCircle; }

    //========================================
    //!\brief Get the vehicleFrontToFrontAxle of the vehicle state in m
    //!\return vehicleFrontToFrontAxle of the vehicle state
    //----------------------------------------
    float getVehicleFrontToFrontAxle() const { return m_vehicleFrontToFrontAxle; }

    //========================================
    //!\brief Get the frontAxleToRearAxle of the vehicle state in m
    //!\return frontAxleToRearAxle of the vehicle state
    //----------------------------------------
    float getFrontAxleToRearAxle() const { return m_frontAxleToRearAxle; }

    //========================================
    //!\brief Get the rearAxleToVehicleRear of the vehicle state in m
    //!\return rearAxleToVehicleRear of the vehicle state
    //----------------------------------------
    float getRearAxleToVehicleRear() const { return m_rearAxleToVehicleRear; }

    //========================================
    //!\brief Get the rearAxleToOrigin of the vehicle state in m
    //!\return rearAxleToOrigin of the vehicle state
    //----------------------------------------
    float getRearAxleToOrigin() const { return m_rearAxleToOrigin; }

    //========================================
    //!\brief Get the total distance driven by the vehicle
    //!\return distance of the vehicle state
    //----------------------------------------
    double getTotalDistanceDriven() const { return m_totalDistanceDriven; }

    //========================================
    //!\brief Get the steering angle of the vehicle state in rad
    //!\return Steering angle of the vehicle state
    //----------------------------------------
    float getSteerAngle() const { return m_steerAngle; }

    //========================================
    //!\brief Get the steering wheel angle of the vehicle state in rad
    //!\return Steering wheel angle of the vehicle state
    //----------------------------------------
    float getSteeringWheelAngle() const { return m_steeringWheelAngle; }

    //========================================
    //!\brief Get the steer ratio coefficient 0 of the vehicle state
    //!\return Steer ratio coefficient 0 of the vehicle state
    //----------------------------------------
    float getSteerRatioCoeff0() const { return m_steerRatioCoeff0; }

    //========================================
    //!\brief Get the steer ratio coefficient 1 of the vehicle state
    //!\return Steer ratio coefficient 1 of the vehicle state
    //----------------------------------------
    float getSteerRatioCoeff1() const { return m_steerRatioCoeff1; }

    //========================================
    //!\brief Get the steer ratio coefficient 2 of the vehicle state
    //!\return Steer ratio coefficient 2 of the vehicle state
    //----------------------------------------
    float getSteerRatioCoeff2() const { return m_steerRatioCoeff2; }

    //========================================
    //!\brief Get the steer ratio coefficient 3 of the vehicle state
    //!\return Steer ratio coefficient 3 of the vehicle state
    //----------------------------------------
    float getSteerRatioCoeff3() const { return m_steerRatioCoeff3; }

public: // setter
    //========================================
    //!\brief Set the microseconds since startup
    //!\param[in] microseconds  New value for microseconds since startup
    //----------------------------------------
    void setMicrosecondsSinceStartup(const uint32_t microseconds) { m_microsecondsSinceStartup = microseconds; }

    //========================================
    //!\brief Set the timestamp of the vehicle state
    //!\param[in] timestamp  New timestamp of the vehicle state
    //----------------------------------------
    void setTimestamp(const NtpTime timestamp) { m_timestamp = timestamp; }

    //========================================
    //!\brief Set the sources of the vehicle state
    //!\param[in] sources  New sources of the vehicle state (bitmask)
    //----------------------------------------
    void setSources(const VehicleStateSource sources) { m_sources = sources; }

    //========================================
    //!\brief Set the blind prediction age of the vehicle state
    //!\param[in] blindPredictionAge  New blind prediction age of the vehicle state
    //----------------------------------------
    void setBlindPredictionAge(const uint16_t blindPredictionAge) { m_blindPredictionAge = blindPredictionAge; }

    //========================================
    //!\brief Set the validation of the vehicle state
    //!\param[in] validation  New validation of the vehicle state (bitmask)
    //----------------------------------------
    void setValidation(const VehicleStateValidation validation) { m_validation = validation; }

    //========================================
    //!\brief Set the coordinateSystem type of the vehicle state
    //!\param[in] coordinateSystem  New coordinateSystem type of the vehicle state (bitmask)
    //----------------------------------------
    void setCoordinateSystem(const VehicleStateCoordinateSystem coordinateSystem)
    {
        m_coordinateSystem = coordinateSystem;
    }

    //========================================
    //!\brief Set the origin position stored in WGS84 of the vehicle state
    //!\param[in] origin New origin position of the vehicle state
    //----------------------------------------
    void setOriginWgs84(const PositionWgs84& origin) { m_originWgs84 = origin; }

    //========================================
    //!\brief Set the origin position stored in Utm of the vehicle state
    //!\param[in] origin New origin position of the vehicle state
    //----------------------------------------
    void setOriginUtm(const PositionUtm& origin) { m_originUtm = origin; }

    //========================================
    //!\brief Set the origin position stored in an internal map of the vehicle state
    //!\param[in] origin New origin position of the vehicle state
    //----------------------------------------
    void setOriginInternalMap(const std::string& origin) { m_originInternalMap = origin; }

    //========================================
    //!\brief Set the actual position  of the vehicle state
    //!\param[in] position New position of the vehicle state
    //----------------------------------------
    void setPosition(const Vector3<double>& position) { m_position = position; }

    //========================================
    //!\brief Set the orientation of the vehicle state as Euler vector
    //!\param[in] orientation New orientation of the vehicle state with the rotation order yaw, pitch, roll
    //----------------------------------------
    void setOrientationAsVector(const Vector3<double>& orientation)
    {
        m_orientationAsRotationMatrix.setFromVectorWithRotationOrderRollPitchYaw(orientation);
    }

    //========================================
    //!\brief Set the orientation of the vehicle state as rotation matrix
    //!\param[in] orientation New orientation of the vehicle state
    //----------------------------------------
    void setOrientationAsRotationMatrix(const RotationMatrix3d<double>& orientation)
    {
        m_orientationAsRotationMatrix = orientation;
    }

    //========================================
    //!\brief Set the orientation of the vehicle state as quaternion
    //!\param[in] orientation New orientation of the vehicle state
    //----------------------------------------
    void setOrientationAsQuaternion(const Quaternion<double>& orientation)
    {
        m_orientationAsRotationMatrix = orientation.getRotationMatrix();
    }

    //========================================
    //!\brief Set the actual position and orientation as Euler vector sigma of the vehicle state
    //!\param[in] sigma New sigma of the vehicle state
    //----------------------------------------
    void setPositionOrientationSigma(const Matrix6x6<double>& sigma) { m_positionOrientationSigma = sigma; }

    //========================================
    //!\brief Set the course angle of the vehicle state
    //!\param[in] courseAngle New course angle of the vehicle state
    //----------------------------------------
    void setCourseAngle(const float courseAngle) { m_courseAngle = courseAngle; }

    //========================================
    //!\brief Set the sigma of the course angle of the vehicle state
    //!\param[in] courseAngleSigma New course angle sigma of the vehicle state
    //----------------------------------------
    void setCourseAngleSigma(const float courseAngleSigma) { m_courseAngleSigma = courseAngleSigma; }

    //========================================
    //!\brief Set the linear velocity of the vehicle state in m/s
    //!\param[in] velocity New velocity  of the vehicle state
    //----------------------------------------
    void setLinearVelocity(const Vector3<float>& velocity) { m_linearVelocity = velocity; }

    //========================================
    //!\brief Set the sigma of the linear velocity of the vehicle state in  m/s
    //!\param[in] sigma New velocity sigma of the vehicle state
    //----------------------------------------
    void setLinearVelocitySigma(const Matrix3x3<float>& sigma) { m_linearVelocitySigma = sigma; }

    //========================================
    //!\brief Set the angular velocity of the vehicle state in rad/s
    //!\param[in] velocity New velocity  of the vehicle state
    //----------------------------------------
    void setAngularVelocity(const Vector3<float>& velocity) { m_angularVelocity = velocity; }

    //========================================
    //!\brief Set the sigma of the angular velocity of the vehicle state in  rad/s
    //!\param[in] sigma  New velocity sigma of the vehicle state
    //----------------------------------------
    void setAngularVelocitySigma(const Matrix3x3<float>& sigma) { m_angularVelocitySigma = sigma; }

    //========================================
    //!\brief Set the linear acceleration of the vehicle state in  m/s^2
    //!\param[in] acceleration New acceleration of the vehicle state
    //----------------------------------------
    void setAcceleration(const Vector3<float>& acceleration) { m_acceleration = acceleration; }

    //========================================
    //!\brief Set the sigma of the linear acceleration of the vehicle state in  m/s^2
    //!\param[in] sigma New acceleration sigma of the vehicle state
    //----------------------------------------
    void setAccelerationSigma(const Matrix3x3<float>& sigma) { m_accelerationSigma = sigma; }

    //========================================
    //!\brief Set the vehicle body to road surface position of the vehicle state in  m
    //!\param[in] position  New vehicle body to road surface position of the vehicle state
    //----------------------------------------
    void setVehicleBodyToRoadSurfacePosition(const Vector3<double>& position)
    {
        m_vehicleBodyToRoadSurfacePosition = position;
    }

    //========================================
    //!\brief Set the vehicle body to road surface orientation of the vehicle state in rad as Euler vector
    //!\param[in] orientation New vehicle body to road surface orientation of
    //!           the vehicle state with the rotation order yaw, pitch, roll
    //----------------------------------------
    void setVehicleBodyToRoadSurfaceOrientationAsVector(const Vector3<double>& orientation)
    {
        m_vehicleBodyToRoadSurfaceOrientationAsRotationMatrix.setFromVectorWithRotationOrderRollPitchYaw(orientation);
    }

    //========================================
    //!\brief Set the vehicle body to road surface orientation of the vehicle state as rotation matrix
    //!\param[in]  orientation New vehicle body to road surface orientation of the vehicle state
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
    //!\brief Set the sigma of the vehicle body to road surface position and orientation of the vehicle state
    //!\param[in]  sigma New sigma of the vehicle body to road surface position and orientation of the vehicle state
    //----------------------------------------
    void setVehicleBodyToRoadSurfacePositionOrientationSigma(const Matrix6x6<double>& sigma)
    {
        m_vehicleBodyToRoadSurfacePositionOrientationSigma = sigma;
    }

    //========================================
    //!\brief Set the vehicle width of the vehicle state in m
    //!\param[in] vehicleWidth New vehicle width of the vehicle state
    //----------------------------------------
    void setVehicleWidth(const float vehicleWidth) { m_vehicleWidth = vehicleWidth; }

    //========================================
    //!\brief Set the min turning cicle of the vehicle state in m
    //!\param[in]  minTurningCircle New min turning circle of the vehicle state
    //----------------------------------------
    void setMinTurningCircle(const float minTurningCircle) { m_minTurningCircle = minTurningCircle; }

    //========================================
    //!\brief Set the distance from the vehicle front to front axle of the vehicle state in m
    //!\param[in] vehicleFrontToFrontAxle New distance
    //----------------------------------------
    void setVehicleFrontToFrontAxle(const float vehicleFrontToFrontAxle)
    {
        m_vehicleFrontToFrontAxle = vehicleFrontToFrontAxle;
    }

    //========================================
    //!\brief Set the distance from the front axle to rear axle of the vehicle state in m
    //!\param[in] frontAxleToRearAxle New distance
    //----------------------------------------
    void setFrontAxleToRearAxle(const float frontAxleToRearAxle) { m_frontAxleToRearAxle = frontAxleToRearAxle; }

    //========================================
    //!\brief Set the distance from the rear axle to vehicle rear of the vehicle state in m
    //!\param[in] rearAxleToVehicleRear New distance
    //----------------------------------------
    void setRearAxleToVehicleRear(const float rearAxleToVehicleRear)
    {
        m_rearAxleToVehicleRear = rearAxleToVehicleRear;
    }

    //========================================
    //!\brief Set the distance from the rear axle to origin of the vehicle state in m
    //!\param[in] rearAxleToOrigin New distance
    //----------------------------------------
    void setRearAxleToOrigin(const float rearAxleToOrigin) { m_rearAxleToOrigin = rearAxleToOrigin; }

    //========================================
    //!\brief Set the distance by the vehicle
    //!\param[in] distance New distance
    //----------------------------------------
    void setTotalDistanceDriven(const double distance) { m_totalDistanceDriven = distance; }

    //========================================
    //!\brief Set the steer angle of the vehicle state in rad
    //!\param[in] steerAngle New steer angle of the vehicle state
    //----------------------------------------
    void setSteerAngle(const float steerAngle) { m_steerAngle = steerAngle; }

    //========================================
    //!\brief Set the steering wheel angle of the vehicle state in rad
    //!\param[in] steeringWheelAngle New steeringWheelAngle of the vehicle state
    //----------------------------------------
    void setSteeringWheelAngle(const float steeringWheelAngle) { m_steeringWheelAngle = steeringWheelAngle; }

    //========================================
    //!\brief Set the steer ration coefficient 0 of the vehicle state
    //!\param[in] steerRatioCoeff0 New coefficient of the vehicle state
    //----------------------------------------
    void setSteerRatioCoeff0(const float steerRatioCoeff0) { m_steerRatioCoeff0 = steerRatioCoeff0; }

    //========================================
    //!\brief Set the steer ration coefficient 1 of the vehicle state
    //!\param[in] steerRatioCoeff1 New coefficient of the vehicle state
    //----------------------------------------
    void setSteerRatioCoeff1(const float steerRatioCoeff1) { m_steerRatioCoeff1 = steerRatioCoeff1; }

    //========================================
    //!\brief Set the steer ration coefficient 2 of the vehicle state
    //!\param[in] steerRatioCoeff2 New coefficient of the vehicle state
    //----------------------------------------
    void setSteerRatioCoeff2(const float steerRatioCoeff2) { m_steerRatioCoeff2 = steerRatioCoeff2; }

    //========================================
    //!\brief Set the steer ration coefficient 3 of the vehicle state
    //!\param[in] steerRatioCoeff3 New coefficient of the vehicles tate
    //----------------------------------------
    void setSteerRatioCoeff3(const float steerRatioCoeff3) { m_steerRatioCoeff3 = steerRatioCoeff3; }

protected:
    //!@{
    //! Time members
    uint32_t m_microsecondsSinceStartup{0}; //!< microseconds since startup [µs]
    NtpTime m_timestamp{}; //!< timestamp of this data
    uint16_t m_blindPredictionAge{0}; //!< Number of scans since extrapolation of the vehicle state without an update
    //!@}

    //!@{
    //! Flags
    VehicleStateSource m_sources{VehicleStateSource::Unknown}; //!< Who delivered the input (Bitmask)
    VehicleStateValidation m_validation{VehicleStateValidation::Ok}; //!< The validation status (Bitmask)
    //!@}

    //!@{
    //! Origin position global
    VehicleStateCoordinateSystem m_coordinateSystem{
        VehicleStateCoordinateSystem::Unknown}; //!< coordinate system for this frame (Bitmask)
    PositionWgs84 m_originWgs84; //!< reference point for GlobalTangentialPlane
    PositionUtm m_originUtm; //!< Utm coordinates
    std::string m_originInternalMap; //!< Map name for MapCoordinates
    //!@}

    //!@{
    //! 3D position and orientation of vehicle (VRC)
    Vector3<double> m_position; //!< Absolute Position from origin [m] (x,y,z)
    RotationMatrix3d<double> m_orientationAsRotationMatrix; //!< current vehicle angle
    Matrix6x6<double> m_positionOrientationSigma; //!< standard deviation [???] (x,y,z,roll,pitch,yaw)

    float m_courseAngle{NaN}; //!< Absolute orientation at time timeStamp [rad]
    float m_courseAngleSigma{NaN}; //!< standard deviation [rad]
    //!@}

    //!@{
    //! 3D Velocity (VRC)
    Vector3<float> m_linearVelocity; //!< Current velocity of the vehicle [m/s] (x,y,z)
    Matrix3x3<float> m_linearVelocitySigma; //!< standard deviation [m/s] (x,y,z)
    Vector3<float> m_angularVelocity; //!< Current angular velocity of the vehicle [rad/s]   (roll,pitch,yaw)
    Matrix3x3<float> m_angularVelocitySigma; //!< standard deviation [rad/s]  (roll,pitch,yaw)
    //!@}

    //!@{
    //! 3D Linear Acceleration (VRC)
    Vector3<float> m_acceleration; //!< Current acceleration of the vehicle [m/s^2] (x,y,z)
    Matrix3x3<float> m_accelerationSigma; //!< standard deviation [m/s^2] (x,y,z)
    //!@}

    //!@{
    //! Vehicle Body To Road Surface (VBC-to-VRC) VRC= VehicleRoad coordinate system ; VBC=VehicleBody coordinate system
    Vector3<double> m_vehicleBodyToRoadSurfacePosition; //!< Distance of vehicle body to the road surface [m] (x,y,z)
    RotationMatrix3d<double>
        m_vehicleBodyToRoadSurfaceOrientationAsRotationMatrix; //!<  Orientation of vehicle body to the road surface
    Matrix6x6<double>
        m_vehicleBodyToRoadSurfacePositionOrientationSigma; //!< standard deviation [???] (x,y,z,roll,pitch,yaw)

    //!@}

    //!@{
    //! Distances
    float m_vehicleFrontToFrontAxle{NaN}; //!< Distance: vehicle's front axle to vehicle's front [m]
    float m_frontAxleToRearAxle{NaN}; //!< Distance: vehicle's rear axle to vehicle's front axle [m]
    float m_rearAxleToVehicleRear{NaN}; //!< Distance: vehicle's rear axle to vehicle's rear [m]
    float m_rearAxleToOrigin{NaN}; //!< Distance: vehicle's rear axle to origin [m]
    double m_totalDistanceDriven{0}; //!< Distance: total distance driven [m]
    float m_vehicleWidth{NaN}; //!< Vehicle width [m]
    float m_minTurningCircle{NaN}; //!< Minimal turning circle [m]
    //!@}

    //!@{
    //! Steering
    float m_steerAngle{NaN}; //!< [rad]
    float m_steeringWheelAngle{NaN}; //!< steering wheel angle [rad]
    float m_steerRatioCoeff0{NaN}; //!< s_3*x^3 + s_2*x^2 + s-2*x^1 + s_0
    float m_steerRatioCoeff1{NaN}; //!< s_3*x^3 + s_2*x^2 + s-2*x^1 + s_0
    float m_steerRatioCoeff2{NaN}; //!< s_3*x^3 + s_2*x^2 + s-2*x^1 + s_0
    float m_steerRatioCoeff3{NaN}; //!< s_3*x^3 + s_2*x^2 + s-2*x^1 + s_0
    //!@}

}; // VehicleState2809

//==============================================================================

//==============================================================================

bool operator==(const VehicleState2809& lhs, const VehicleState2809& rhs);
bool operator!=(const VehicleState2809& lhs, const VehicleState2809& rhs);

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
