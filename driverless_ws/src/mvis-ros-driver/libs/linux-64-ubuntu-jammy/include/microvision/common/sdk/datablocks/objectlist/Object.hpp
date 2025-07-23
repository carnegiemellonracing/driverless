//==============================================================================
//! \file
//!
//! \brief General object with members in SI units (m, rad, m/s, rad/s).
//!
//! $$MICROVISION_LICENSE_BEGIN$$
//! Copyright (c) 2025 MicroVision, Inc., Redmond, U.S.A.
//! All Rights Reserved.
//!
//! For more details, please refer to the accompanying file
//! License.txt.
//! $$MICROVISION_LICENSE_END$$
//!
//! \date Apr 12, 2018
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/objectlist/ContourPoint.hpp>
#include <microvision/common/sdk/datablocks/objectlist/UnfilteredObjectData.hpp>
#include <microvision/common/sdk/datablocks/objectlist/special/ObjectIn2281.hpp>
#include <microvision/common/sdk/datablocks/RegisteredImporter.hpp>
#include <microvision/common/sdk/datablocks/MeasurementSeries.hpp>
#include <microvision/common/sdk/Vector2.hpp>
#include <microvision/common/sdk/DataTypeId.hpp>
#include <microvision/common/sdk/ObjectBasic.hpp>
#include <microvision/common/sdk/NtpTime.hpp>

#include <stddef.h>
#include <cstdint>
#include <iostream>
#include <vector>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

class Object final
{
    template<class ContainerType, DataTypeId::DataType id>
    friend class Importer;
    template<class ContainerType, DataTypeId::DataType id>
    friend class Exporter;
    friend class ObjectListImporter2280_2290;
    friend class ObjectListImporter2281_2291;
    friend class ObjectListExporter2281_2291;

public:
    // Object flags.
    static const uint16_t objectFlagTrackedByStationaryModel{0x0040U}; //!< Object is tracked using stationary model.
    static const uint16_t objectFlagMobile{
        0x0080U}; //!< Object has been detected/validated as mobile (the current tracking model is irrelevant;
    //!< this flag just means it has been moving at some time).
    static const uint16_t objectFlagValidated{
        0x0100U}; //!< Object (stationary or dynamic) has been validated, i.e. valid enough to send out to the interface.
    static const uint16_t objectFlagOccluded{
        0x0200U}; //!< Object was partly or fully occluded by another object or the edge of the scan field of
    //!< view in the last update cycle, such that the full extent of the object was not visible.
    static const uint16_t objectFlagOverdrivable{
        0x0400U}; //!< Object is low and can likely be driven over (curb, speed bump).
    static const uint16_t objectFlagUnderdrivable{
        0x0800U}; //!< Object is suspended above the street and can likely be driven under
    //!< (sign, bridge, parking garage).

public:
    //========================================
    //! \brief Mask for flag nibble in #m_dynamicFlags.
    //!
    //! The 4 least significant bits of #m_dynamicFlags are reserved for
    //! for flags. Only bit 0 is currently used.
    //----------------------------------------
    static constexpr uint8_t dynamicFlagMask = 0xFU;

    //========================================
    //! \brief Indicates existence.
    //!
    //! - 0: init
    //! - 1: exisitng
    //----------------------------------------
    static constexpr uint8_t dynamicFlagExisting = 0x1U;

    //========================================
    //! The 4 bits of most significant bits represents an enum for the dynamic state.
    //! Currently only 3 bits (4, 5 and 6) are used.
    //----------------------------------------
    static constexpr uint8_t dynamicStateMask{static_cast<uint8_t>(0xFU << 4)};

    //========================================
    //!\name DynamicStates Dynamic states
    //! There are four dynamic states known and also the unknown dynamic state.
    //!@{
    static constexpr uint8_t dynamicStateUnknown{static_cast<uint8_t>(0 << 4)};
    static constexpr uint8_t dynamicStateMoving{static_cast<uint8_t>(1 << 4)};
    static constexpr uint8_t dynamicStateStopped{static_cast<uint8_t>(2 << 4)};
    static constexpr uint8_t dynamicStateStarted{static_cast<uint8_t>(3 << 4)};
    static constexpr uint8_t dynamicStateAPrioriStatic{static_cast<uint8_t>(4 << 4)};
    static constexpr uint8_t dynamicStateStaticInit{static_cast<uint8_t>(5 << 4)};
    static constexpr uint8_t dynamicStateDynamicInit{static_cast<uint8_t>(6 << 4)};
    //!@}

public:
    static Vector2<float> getObjectBoxPosition(RefPointBoxLocation curRefPtLoc,
                                               const Vector2<float> refPt,
                                               const float courseAngle,
                                               const Vector2<float> objBoxSize,
                                               RefPointBoxLocation targetRefPtLoc)
    {
        return ObjectIn2281::getObjectBoxPosition(curRefPtLoc, refPt, courseAngle, objBoxSize, targetRefPtLoc);
    }

public:
    Object()          = default;
    virtual ~Object() = default;

public:
    void addMeasurement(const Measurement& meas) { m_measurementList.addMeasurement(meas); }
    void addContourPoint(const ContourPoint& contourPoint) { m_contourPoints.push_back(contourPoint); }

public: // getter
    uint32_t getObjectId() const { return m_objectId; }
    NtpTime getTimestamp() const { return m_timestamp; }

    uint16_t getObjectFlags() const { return m_objectFlags; }

public:
    //========================================
    //! \name ObjectFlags
    //!
    //! Methods to test for object flags to be set.
    //!
    //! @{
    //----------------------------------------
    bool isTrackedByStationaryModel() const
    {
        return ((getObjectFlags() & objectFlagTrackedByStationaryModel) == objectFlagTrackedByStationaryModel);
    }
    bool isMobile() const { return ((getObjectFlags() & objectFlagMobile) == objectFlagMobile); }
    bool isValidated() const { return ((getObjectFlags() & objectFlagValidated) == objectFlagValidated); }
    bool isOccluded() const { return ((getObjectFlags() & objectFlagOccluded) != objectFlagOccluded); }
    bool isOverdrivable() const { return ((getObjectFlags() & objectFlagOverdrivable) == objectFlagOverdrivable); }
    bool isUnderdrivable() const { return ((getObjectFlags() & objectFlagUnderdrivable) == objectFlagUnderdrivable); }
    //========================================
    //! @}
    //----------------------------------------

public:
    //========================================
    //! \name DynamicFlags
    //!
    //! The 4 least significant nibble of #m_dynamicFlags is reserved for
    //! for flags. Only bit 0 is currently used.
    //!
    //! @{
    //----------------------------------------

    //========================================
    //! \brief Get the value of the #m_dynamicFlags variable,
    //!        of both, the flags and and the dynamic state, part.
    //! \return The content of the #m_dynamicFlags variable.
    //----------------------------------------
    uint8_t getDynamicFlagsAndState() const { return m_dynamicFlags; }

    //========================================
    //! \brief Get the flags part of the #m_dynamicFlags variable.
    //! \return The flags part of the #m_dynamicFlags variables.
    //----------------------------------------
    uint8_t getDynamicFlags() const { return (m_dynamicFlags & dynamicFlagMask); }

    //========================================
    //! \brief Check, whether the given dynamic flag(s) is(are) set.
    //! \param[in] flagsToTest  The flag to be tested. \a flagToTest
    //!                         is masked by #dynamicFlagMask.
    //! \return \c True if all given flags are set, \c false otherwise.
    //----------------------------------------
    bool areDynamicFlagsSet(const uint8_t flagsToTest) const
    {
        return testBitsInDynamicFlags(static_cast<uint8_t>(flagsToTest & dynamicFlagMask));
    }

    //========================================
    //! \brief Checks whether the flag dynamicFlagExisting in m_dynamicFlags set
    //!        to 1 (existing).
    //! \return \c True if the object exists,
    //!         \c false if the object is still in init state.
    //----------------------------------------
    bool isExisting() const { return ((getDynamicFlags() & dynamicFlagExisting) == dynamicFlagExisting); }
    //========================================
    //! @}
    //----------------------------------------

public:
    //========================================
    //! \name DynamicStates
    //!
    //! The dynamic state is stored in the most significant nibble of m_dynamicFlags
    //! as an enum. Currently only 3 bits (4, 5 and 6) are used.
    //!
    //! @{
    //----------------------------------------

    //========================================
    //! \brief Return the dynamic state.
    //!
    //! \return The dynamic state.
    //----------------------------------------
    uint8_t getDynamicState() const { return static_cast<uint8_t>(m_dynamicFlags & dynamicStateMask); }

    //========================================
    //! \brief Returns whether the dynamic state is unknown.
    //! \return \c True if the dynamic state is #dynamicStateUnknown.
    //!         \c false otherwise.
    //----------------------------------------
    bool isDynamicStateUnknown() const { return (getDynamicState() == dynamicStateUnknown); }

    //========================================
    //! \brief Returns whether the dynamic state is set to moving.
    //! \return \c True if the dynamic state is set to #dynamicStateMoving,
    //!         \c false otherwise.
    //----------------------------------------
    bool isDynamicStateMoving() const { return (getDynamicState() == dynamicStateMoving); }

    //========================================
    //! \brief Returns whether the dynamic state is set to stopped.
    //! \return \c True if the dynamic state is set to #dynamicStateStopped,
    //!         \c false otherwise.
    //----------------------------------------
    bool isDynamicStateStopped() const { return (getDynamicState() == dynamicStateStopped); }

    //========================================
    //! \brief Returns whether the dynamic state is set to started.
    //! \return \c True if the dynamic state is set to #dynamicStateStarted,
    //!         \c false otherwise.
    //----------------------------------------
    bool isDynamicStateStarted() const { return (getDynamicState() == dynamicStateStarted); }

    //========================================
    //! \brief Returns whether the dynamic state is set to a priory static.
    //! \return \c True if the dynamic state is set to #dynamicStateAPriorityStatic,
    //!         \c false otherwise.
    //----------------------------------------
    bool isDynamicStateAPrioriStatic() const { return (getDynamicState() == dynamicStateAPrioriStatic); }

    //========================================
    //! \brief Returns whether the dynamic state is set to started.
    //! \return \c True if the dynamic state is set to #dynamicStateStarted,
    //!         \c false otherwise.
    //----------------------------------------
    bool isStaticInit() const { return (getDynamicState() == dynamicStateStaticInit); }

    //========================================
    //! \brief Returns whether the dynamic state is set to started.
    //! \return \c True if the dynamic state is set to #dynamicStateStarted,
    //!         \c false otherwise.
    //----------------------------------------
    bool isDynamicInit() const { return (getDynamicState() == dynamicStateDynamicInit); }

    //========================================
    //! @}
    //----------------------------------------

public:
    uint32_t getObjectAge() const { return m_objectAge; }
    uint16_t getHiddenStatusAge() const { return m_hiddenStatusAge; }

    Vector2<float> getObjectBoxSize() const { return m_objectBoxSize; }
    Vector2<float> getObjectBoxSizeSigma() const { return m_objectBoxSizeSigma; }
    float getObjectBoxHeight() const { return m_objectBoxHeight; }
    float getObjectBoxHeightSigma() const { return m_objectBoxHeightSigma; }
    float getObjectBoxHeightOffset() const { return m_objectBoxHeightOffset; }
    float getObjectBoxHeightOffsetSigma() const { return m_objectBoxHeightOffsetSigma; }

    RefPointBoxLocation getReferencePointLocation() const { return m_referencePointLocation; }
    Vector2<float> getReferencePointCoord() const { return m_referencePointCoord; }
    Vector2<float> getReferencePointCoordSigma() const { return m_referencePointCoordSigma; }
    float getReferencePointCoordCorrCoeff() const { return m_referencePointCoordCorrCoeff; }

    float getCourseAngle() const { return m_courseAngle; }
    float getCourseAngleSigma() const { return m_courseAngleSigma; }
    float getYawRate() const { return m_yawRate; }
    float getYawRateSigma() const { return m_yawRateSigma; }

    ObjectClass getClassification() const { return m_classification; }
    float getClassificationQuality() const { return m_classificationQuality; }
    uint32_t getClassificationAge() const { return m_classificationAge; }

    Vector2<float> getMotionReferencePoint() const { return m_motionReferencePoint; }
    Vector2<float> getMotionReferencePointSigma() const { return m_motionReferencePointSigma; }

    Vector2<float> getCenterOfGravity() const { return m_centerOfGravity; }
    float getExistenceProbability() const { return m_existenceProbability; }
    uint16_t getObjectPriority() const { return m_objectPriority; }

    Vector2<float> getRelativeVelocity() const { return m_relativeVelocity; }
    Vector2<float> getRelativeVelocitySigma() const { return m_relativeVelocitySigma; }
    Vector2<float> getAbsoluteVelocity() const { return m_absoluteVelocity; }
    Vector2<float> getAbsoluteVelocitySigma() const { return m_absoluteVelocitySigma; }
    float getAbsoluteVelocityCorrCoeff() const;

    Vector2<float> getAcceleration() const { return m_acceleration; }
    Vector2<float> getAccelerationSigma() const { return m_accelerationSigma; }
    float getAccelerationCorrCoeff() const { return m_accelerationCorrCoeff; }
    float getLongitudinalAcceleration() const;
    float getLongitudinalAccelerationSigma() const;

    Vector2<float> getClosestObjectPointCoord() const { return m_closestObjectPointCoord; }
    uint8_t getNumContourPoints() const { return static_cast<uint8_t>(m_contourPoints.size()); }
    const std::vector<ContourPoint>& getContourPoints() const { return m_contourPoints; }
    std::vector<ContourPoint>& getContourPoints() { return m_contourPoints; }

    const MeasurementSeries& getMeasurements() const { return m_measurementList; }
    MeasurementSeries& getMeasurements() { return m_measurementList; }
    const MeasurementSeries& dynamicObjectProperties() const { return getMeasurements(); }
    MeasurementSeries& dynamicObjectProperties() { return getMeasurements(); }

    bool hasUnfilteredObjectData() const { return (m_unfilteredObjectData != nullptr); }
    const UnfilteredObjectData* getUnfilteredObjectData() const { return m_unfilteredObjectData; }
    UnfilteredObjectData* getUnfilteredObjectData() { return m_unfilteredObjectData; }

    Vector2<float> getObjectBoxPosition(const RefPointBoxLocation targetRefPointLocation) const;
    Vector2<float> getCenterPoint() const { return getObjectBoxPosition(RefPointBoxLocation::ObjectCenter); }

public: // setter
    void setObjectId(const uint32_t value) { m_objectId = value; }
    void setTimestamp(const NtpTime timestamp) { m_timestamp = timestamp; }

    void setObjectFlags(const uint16_t objectFlags) { m_objectFlags = objectFlags; }
    void setTrackedByStationaryModel(const bool set = true) { setObjectFlag(objectFlagTrackedByStationaryModel, set); }
    void setMobile(const bool set = true) { setObjectFlag(objectFlagMobile, set); }
    void setValidated(const bool set = true) { setObjectFlag(objectFlagValidated, set); }
    void setOccluded(const bool set = true) { setObjectFlag(objectFlagOccluded, set); }
    void setOverdrivable(const bool set = true) { setObjectFlag(objectFlagOverdrivable, set); }
    void setUnderdrivable(const bool set = true) { setObjectFlag(objectFlagUnderdrivable, set); }

public:
    //========================================
    //! \name DynamicFlags
    //!
    //! The 4 least significant nibble of #m_dynamicFlags is reserved for
    //! for flags. Only bit 0 is currently used.
    //!
    //! @{
    //----------------------------------------

    //========================================
    //! \brief Set the complete dynamic flag variable, i.e. flags
    //!        and dynamic state part.
    //! \param[in] newDynamicFlags  The new value for the dynamic flags variable.
    //----------------------------------------
    void setDynamicFlagsAndState(const uint8_t newDynamicFlagsAndState) { m_dynamicFlags = newDynamicFlagsAndState; }

    //========================================
    //! \brief Set the existing flag in #m_dynamicFlags.
    //----------------------------------------
    void setIsExistingFlag() { m_dynamicFlags = m_dynamicFlags | dynamicFlagExisting; }

    //========================================
    //! \brief Clear the existing flag in #m_dynamicFlags.
    //----------------------------------------
    void clearIsExistingFlag() { m_dynamicFlags = static_cast<uint8_t>(m_dynamicFlags & (~dynamicFlagExisting)); }

    //========================================
    //! @}
    //----------------------------------------

public:
    //========================================
    //! \name DynamicStates
    //!
    //! The dynamic state is stored in the most significant nibble of m_dynamicFlags
    //! as an enum. Currently only 3 bits (4, 5 and 6) are used.
    //!
    //! @{
    //----------------------------------------

    //========================================
    //! \brief Set the dynamic state to unknown.
    //----------------------------------------
    void setDynamicStateUnknown() { setDynamicState(dynamicStateUnknown); }

    //========================================
    //! \brief Set the dynamic state to moving.
    //----------------------------------------
    void setDynamicStateMoving() { setDynamicState(dynamicStateMoving); }

    //========================================
    //! \brief Set the dynamic state to stopped.
    //----------------------------------------
    void setDynamicStateStopped() { setDynamicState(dynamicStateStopped); }

    //========================================
    //! \brief Set the dynamic state to started.
    //----------------------------------------
    void setDynamicStateStarted() { setDynamicState(dynamicStateStarted); }

    //========================================
    //! \brief Set the dynamic state to a priory static.
    //----------------------------------------
    void setDynamicStateAPrioriStatic() { setDynamicState(dynamicStateAPrioriStatic); }

    //========================================
    //! \brief Set the dynamic state to static-init.
    //----------------------------------------
    void setDynamicStateStaticInit() { setDynamicState(dynamicStateStaticInit); }

    //========================================
    //! \brief Set the dynamic state to dynamic-init.
    //----------------------------------------
    void setDynamicStateDynamicInit() { setDynamicState(dynamicStateDynamicInit); }

    //========================================
    //! @}
    //----------------------------------------

public:
    void setObjectAge(const uint32_t objectAge) { m_objectAge = objectAge; }
    void setHiddenStatusAge(const uint16_t hiddenStatusAge) { m_hiddenStatusAge = hiddenStatusAge; }

    void setObjectBoxSize(const Vector2<float>& objectBoxSize) { m_objectBoxSize = objectBoxSize; }
    void setObjectBoxSizeSigma(const Vector2<float>& objectBoxSizeSigma) { m_objectBoxSizeSigma = objectBoxSizeSigma; }
    void setObjectBoxHeight(const float objectBoxHeight) { m_objectBoxHeight = objectBoxHeight; }
    void setObjectBoxHeightSigma(const float objectBoxHeightSigma) { m_objectBoxHeightSigma = objectBoxHeightSigma; }
    void setObjectBoxHeightOffset(const float objectBoxHeightOffset)
    {
        m_objectBoxHeightOffset = objectBoxHeightOffset;
    }
    void setObjectBoxHeightOffsetSigma(const float objectBoxHeightOffsetSigma)
    {
        m_objectBoxHeightOffsetSigma = objectBoxHeightOffsetSigma;
    }

    void setReferencePointLocation(const RefPointBoxLocation referencePointLocation)
    {
        m_referencePointLocation = referencePointLocation;
    }
    void setReferencePointCoord(const Vector2<float>& referencePointCoord)
    {
        m_referencePointCoord = referencePointCoord;
    }
    void setReferencePointCoordSigma(const Vector2<float>& referencePointCoordSigma)
    {
        m_referencePointCoordSigma = referencePointCoordSigma;
    }
    void setReferencePointCoordCorrCoeff(const float referencePointCoordCorrCoeff)
    {
        m_referencePointCoordCorrCoeff = referencePointCoordCorrCoeff;
    }

    void setCourseAngle(const float courseAngle) { m_courseAngle = courseAngle; }
    void setCourseAngleSigma(const float courseAngleSigma) { m_courseAngleSigma = courseAngleSigma; }
    void setYawRate(const float yawRate) { m_yawRate = yawRate; }
    void setYawRateSigma(const float yawRateSigma) { m_yawRateSigma = yawRateSigma; }

    void setClassification(const ObjectClass classification) { m_classification = classification; }
    void setClassificationQuality(const float classificationQuality)
    {
        m_classificationQuality = classificationQuality;
    }
    void setClassificationAge(const uint32_t classificationAge) { m_classificationAge = classificationAge; }

    void setMotionReferencePoint(const Vector2<float>& motionReferencePoint)
    {
        m_motionReferencePoint = motionReferencePoint;
    }
    void setMotionReferencePointSigma(const Vector2<float>& motionReferencePointSigma)
    {
        m_motionReferencePointSigma = motionReferencePointSigma;
    }

    void setCenterOfGravity(const Vector2<float>& centerOfGravity) { m_centerOfGravity = centerOfGravity; }
    void setExistenceProbability(const float existenceProbability) { m_existenceProbability = existenceProbability; }
    void setObjectPriority(const uint16_t objectPriority) { m_objectPriority = objectPriority; }

    void setRelativeVelocity(const Vector2<float>& relativeVelocity) { m_relativeVelocity = relativeVelocity; }
    void setRelativeVelocitySigma(const Vector2<float>& relativeVelocitySigma)
    {
        m_relativeVelocitySigma = relativeVelocitySigma;
    }
    void setAbsoluteVelocity(const Vector2<float>& absoluteVelocity) { m_absoluteVelocity = absoluteVelocity; }
    void setAbsoluteVelocitySigma(const Vector2<float>& absoluteVelocitySigma)
    {
        m_absoluteVelocitySigma = absoluteVelocitySigma;
    }

    void setAcceleration(const Vector2<float> acceleration) { m_acceleration = acceleration; }
    void setAccelerationSigma(const Vector2<float> accelerationSigma) { m_accelerationSigma = accelerationSigma; }

    void setClosestObjectPointCoord(const Vector2<float>& closestObjectPointCoord)
    {
        m_closestObjectPointCoord = closestObjectPointCoord;
    }
    void setContourPoints(const std::vector<ContourPoint>& contourPoints) { m_contourPoints = contourPoints; }

    void setMeasurementList(const MeasurementSeries& measurementList) { m_measurementList = measurementList; }
    void setDynamicObjectProperties(const MeasurementSeries& dynamicObjectProperties)
    {
        setMeasurementList(dynamicObjectProperties);
    }

    void setUnfilteredObjectData(const UnfilteredObjectData& unfilteredObjectData)
    {
        if (m_unfilteredObjectData == nullptr)
        {
            m_unfilteredObjectData = new UnfilteredObjectData();
        }

        *m_unfilteredObjectData = unfilteredObjectData;
    }

private:
    static std::vector<int> getDirectionVector(const RefPointBoxLocation location);

    void setObjectFlag(const uint16_t flag, const bool set)
    {
        m_objectFlags = static_cast<uint16_t>(set ? (m_objectFlags | flag) : (m_objectFlags & ~flag));
    }

    bool testBitsInDynamicFlags(const uint8_t flagsToTest) const
    {
        return ((m_dynamicFlags & flagsToTest) == flagsToTest);
    }

    //========================================
    //! \brief Set a new dynamic state.
    //!
    //! \param[in] newState  The new value for the new dynamic state.
    //!                      \a newState will be masked by #dynamicStateMask.
    //!                      Only the bits 4 to 7 are considered.
    //----------------------------------------
    void setDynamicState(const uint8_t newState)
    {
        m_dynamicFlags = static_cast<uint8_t>(m_dynamicFlags & (~dynamicStateMask)); // Clear current state.
        m_dynamicFlags = static_cast<uint8_t>(m_dynamicFlags | static_cast<uint8_t>(newState & dynamicStateMask));
    }

    //The following fields MUST be filled before calling this function:
    // - absolute velocity
    // - absolute velocity sigma and correlation coefficient
    // - yaw rate
    // - yaw rate sigma
    void fillAccelerationFields(const float longAcc, const float longAccSigma);

private:
    static constexpr float MinimumVelocityForLongitudinalAcceleration{1e-5F};

    uint32_t m_objectId{0};
    NtpTime m_timestamp{0};

    uint16_t m_objectFlags{0};
    uint8_t m_dynamicFlags{0};
    uint32_t m_objectAge{0};
    uint16_t m_hiddenStatusAge{0};

    Vector2<float> m_objectBoxSize{NaN, NaN};
    Vector2<float> m_objectBoxSizeSigma{NaN, NaN};
    float m_objectBoxHeight{NaN};
    float m_objectBoxHeightSigma{NaN};
    float m_objectBoxHeightOffset{NaN};
    float m_objectBoxHeightOffsetSigma{NaN};

    RefPointBoxLocation m_referencePointLocation{RefPointBoxLocation::Unknown};
    Vector2<float> m_referencePointCoord{NaN, NaN};
    Vector2<float> m_referencePointCoordSigma{NaN, NaN};
    float m_referencePointCoordCorrCoeff{NaN};

    float m_courseAngle{NaN};
    float m_courseAngleSigma{NaN};
    float m_yawRate{NaN};
    float m_yawRateSigma{NaN};

    ObjectClass m_classification{ObjectClass::Unclassified};
    float m_classificationQuality{NaN};
    uint32_t m_classificationAge{0};

    Vector2<float> m_motionReferencePoint{NaN, NaN};
    Vector2<float> m_motionReferencePointSigma{NaN, NaN};

    Vector2<float> m_centerOfGravity{NaN, NaN};
    float m_existenceProbability{NaN};
    uint16_t m_objectPriority{0xFFFFU};

    Vector2<float> m_relativeVelocity{NaN, NaN};
    Vector2<float> m_relativeVelocitySigma{NaN, NaN};
    Vector2<float> m_absoluteVelocity{NaN, NaN};
    Vector2<float> m_absoluteVelocitySigma{NaN, NaN};

    Vector2<float> m_acceleration{NaN, NaN};
    Vector2<float> m_accelerationSigma{NaN, NaN};
    float m_accelerationCorrCoeff{NaN};

    Vector2<float> m_closestObjectPointCoord{NaN, NaN};
    std::vector<ContourPoint> m_contourPoints;
    MeasurementSeries m_measurementList;

    UnfilteredObjectData* m_unfilteredObjectData{nullptr};
}; // Object

//==============================================================================

bool operator==(const Object& v1, const Object& v2);
inline bool operator!=(const Object& v1, const Object& v2) { return !(v1 == v2); }

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
