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
//! \date Apr 23, 2014
//------------------------------------------------------------------------------

//==============================================================================

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/ObjectBasic.hpp>
#include <microvision/common/sdk/datablocks/objectlist/special/ContourPointIn2271.hpp>
#include <microvision/common/sdk/Vector2.hpp>

#include <vector>
#include <iostream>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

class FilteredObjectDataIn2271 final
{
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
    //! - 1: existing
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
    //!
public:
    FilteredObjectDataIn2271()          = default;
    virtual ~FilteredObjectDataIn2271() = default;

public:
    bool isValid() const { return m_isValid; }
    bool hasContourPoints() const { return m_hasContourPoints; }

    uint8_t getPriority() const { return m_priority; }

    uint16_t getObjectAge() const { return m_objectAge; }
    uint16_t getHiddenStatusAge() const { return m_hiddenStatusAge; }

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
    uint16_t getRelativeTimeOfMeasure() const { return m_relativeTimeOfMeasure; }
    Vector2<int16_t> getPositionClosestObjectPoint() const { return m_positionClosestObjectPoint; }

    Vector2<int16_t> getRelativeVelocity() const { return m_relativeVelocity; }
    Vector2<uint16_t> getRelativeVelocitySigma() const { return m_relativeVelocitySigma; }
    rawObjectClass::RawObjectClass getClassification() const { return m_classification; }
    uint8_t getClassificationQuality() const { return m_classificationQuality; }
    uint16_t getClassificationAge() const { return m_classificationAge; }

    uint16_t getReserved() const { return m_reserved; }
    Vector2<uint16_t> getObjectBoxSize() const { return m_objectBoxSize; }
    Vector2<uint16_t> getObjectBoxSizeSigma() const { return m_objectBoxSizeSigma; }
    int16_t getObjectBoxOrientation() const { return m_objectBoxOrientation; }
    uint16_t getObjectBoxOrientationSigma() const { return m_objectBoxOrientationSigma; }
    uint8_t getObjectBoxHeight() const { return m_objectBoxHeight; }
    RefPointBoxLocation getReferencePointLocation() const { return m_referencePointLocation; }
    Vector2<int16_t> getReferencePointCoord() const { return m_referencePointCoord; }
    Vector2<uint16_t> getReferencePointCoordSigma() const { return m_referencePointCoordSigma; }
    int16_t getReferencePointPositionCorrCoeff() const { return m_referencePointPositionCorrCoeff; }
    uint8_t getExistenceProbaility() const { return m_existenceProbaility; }
    uint8_t getPossibleNbOfContourPoints() const { return m_possibleNbOfContourPoints; }

    Vector2<int16_t> getAbsoluteVelocity() const { return m_absoluteVelocity; }
    Vector2<uint16_t> getAbsoluteVelocitySigma() const { return m_absoluteVelocitySigma; }
    int16_t getVelocityCorrCoeff() const { return m_velocityCorrCoeff; }
    Vector2<int16_t> getAcceleration() const { return m_acceleration; }
    Vector2<uint16_t> getAccelerationSigma() const { return m_accelerationSigma; }
    int16_t getAccelerationCorrCoeff() const { return m_accelerationCorrCoeff; }
    int16_t getYawRate() const { return m_yawRate; }
    uint16_t getYawRateSigma() const { return m_yawRateSigma; }

    const std::vector<ContourPointIn2271>& getContourPoints() const { return m_contourPoints; }
    std::vector<ContourPointIn2271>& getContourPoints() { return m_contourPoints; }

public:
    void setIsValid(const bool newIsValid) { m_isValid = newIsValid; }
    void setHasContourPoints(const bool newHasContourPoints) { m_hasContourPoints = newHasContourPoints; }

    void setPriority(const uint8_t newPriority) { m_priority = newPriority; }

    void setObjectAge(const uint16_t newObjectAge) { m_objectAge = newObjectAge; }
    void setHiddenStatusAge(const uint16_t newHiddenStatusAge) { m_hiddenStatusAge = newHiddenStatusAge; }

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
    void setRelativeTimeOfMeasure(const uint16_t newRelativeTimeOfMeasure)
    {
        m_relativeTimeOfMeasure = newRelativeTimeOfMeasure;
    }
    void setPositionClosestObjectPoint(const Vector2<int16_t>& newPositionClosestObjectPoint)
    {
        m_positionClosestObjectPoint = newPositionClosestObjectPoint;
    }

    void setRelativeVelocity(const Vector2<int16_t>& newRelativeVelocity) { m_relativeVelocity = newRelativeVelocity; }
    void setRelativeVelocitySigma(const Vector2<uint16_t>& newRelativeVelocitySigma)
    {
        m_relativeVelocitySigma = newRelativeVelocitySigma;
    }
    void setClassification(const rawObjectClass::RawObjectClass newClassification)
    {
        m_classification = newClassification;
    }
    void setClassificationQuality(const uint8_t newClassificationQuality)
    {
        m_classificationQuality = newClassificationQuality;
    }
    void setClassificationAge(const uint16_t newClassificationAge) { m_classificationAge = newClassificationAge; }

    void setReserved(const uint16_t newReserved) { m_reserved = newReserved; }
    void setObjectBoxSize(const Vector2<uint16_t>& newObjectBoxSize) { m_objectBoxSize = newObjectBoxSize; }
    void setObjectBoxSizeSigma(const Vector2<uint16_t>& newObjectBoxSizeSigma)
    {
        m_objectBoxSizeSigma = newObjectBoxSizeSigma;
    }
    void setObjectBoxOrientation(const int16_t newObjectBoxOrientation)
    {
        m_objectBoxOrientation = newObjectBoxOrientation;
    }
    void setObjectBoxOrientationSigma(const uint16_t newObjectBoxOrientationSigma)
    {
        m_objectBoxOrientationSigma = newObjectBoxOrientationSigma;
    }
    void setObjectBoxHeight(const uint8_t newObjectBoxHeight) { m_objectBoxHeight = newObjectBoxHeight; }
    void setReferencePointLocation(const RefPointBoxLocation newReferencePointLocation)
    {
        m_referencePointLocation = newReferencePointLocation;
    }
    void setReferencePointCoord(const Vector2<int16_t>& newReferencePointCoord)
    {
        m_referencePointCoord = newReferencePointCoord;
    }
    void setReferencePointCoordSigma(const Vector2<uint16_t>& newReferencePointCoordSigma)
    {
        m_referencePointCoordSigma = newReferencePointCoordSigma;
    }
    void setReferencePointPositionCorrCoeff(const int16_t newReferencePointPositionCorrCoeff)
    {
        m_referencePointPositionCorrCoeff = newReferencePointPositionCorrCoeff;
    }
    void setExistenceProbaility(const uint8_t newExistenceProbaility)
    {
        m_existenceProbaility = newExistenceProbaility;
    }

    void setAbsoluteVelocity(const Vector2<int16_t>& newAbsoluteVelocity) { m_absoluteVelocity = newAbsoluteVelocity; }
    void setAbsoluteVelocitySigma(const Vector2<uint16_t>& newAbsoluteVelocitySigma)
    {
        m_absoluteVelocitySigma = newAbsoluteVelocitySigma;
    }
    void setVelocityCorrCoeff(const int16_t newVelocityCorrCoeff) { m_velocityCorrCoeff = newVelocityCorrCoeff; }
    void setAcceleration(const Vector2<int16_t>& newAcceleration) { m_acceleration = newAcceleration; }
    void setAccelerationSigma(const Vector2<uint16_t>& newAccelerationSigma)
    {
        m_accelerationSigma = newAccelerationSigma;
    }
    void setAccelerationCorrCoeff(const int16_t newAccelerationCorrCoeff)
    {
        m_accelerationCorrCoeff = newAccelerationCorrCoeff;
    }
    void setYawRate(const int16_t newWRate) { m_yawRate = newWRate; }
    void setYawRateSigma(const uint16_t newWRateSigma) { m_yawRateSigma = newWRateSigma; }
    // no setter for m_positionClosestObjectPoint, will be adjusted when serializing.

    void setPossibleNbOfContourPoints(const uint8_t possibleNbOfCtPts)
    {
        m_possibleNbOfContourPoints = possibleNbOfCtPts;
    }
    void setContourPoints(const std::vector<ContourPointIn2271>& contourPoints) { m_contourPoints = contourPoints; }

public:
    std::streamsize getSerializedSize() const;
    bool deserialize(std::istream& is);
    bool serialize(std::ostream& os) const;

private:
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

private:
    bool m_isValid{false}; // not serialized
    bool m_hasContourPoints{false}; // not serialized

    uint8_t m_priority{0};

    uint16_t m_objectAge{0};
    uint16_t m_hiddenStatusAge{0};

    //========================================
    //! \brief A combination of dynamic flags
    //!        and the dynamic state.
    //!
    //! The 4 least significant bits are use as flags,
    //! where only bit 0 is currently actually in use:
    //! - #dynamicFlagExistenceValuesMmask
    //!
    //! The 4 most significant bits are used as an enum
    //! to hold the dynamic state:
    //! - #dynamicStateUnknown
    //! - #dynamicStateMoving,
    //! - #dynamicStateStopped
    //! - #dynamicStateStarted,
    //! - #dynamicStateAPriorityStatic
    //----------------------------------------
    uint8_t m_dynamicFlags{0};

    uint16_t m_relativeTimeOfMeasure{0};
    Vector2<int16_t> m_positionClosestObjectPoint{};

    Vector2<int16_t> m_relativeVelocity{};
    Vector2<uint16_t> m_relativeVelocitySigma{};

    rawObjectClass::RawObjectClass m_classification{rawObjectClass::RawObjectClass::Unclassified};
    uint8_t m_classificationQuality{0};
    uint16_t m_classificationAge{0};

    uint16_t m_reserved{0};

    Vector2<uint16_t> m_objectBoxSize{};
    Vector2<uint16_t> m_objectBoxSizeSigma{};
    int16_t m_objectBoxOrientation{0};
    uint16_t m_objectBoxOrientationSigma{0};
    uint8_t m_objectBoxHeight{0};

    RefPointBoxLocation m_referencePointLocation{RefPointBoxLocation::Unknown};
    Vector2<int16_t> m_referencePointCoord{};
    Vector2<uint16_t> m_referencePointCoordSigma{};
    int16_t m_referencePointPositionCorrCoeff{0};

    uint8_t m_existenceProbaility{0};

    Vector2<int16_t> m_absoluteVelocity{};
    Vector2<uint16_t> m_absoluteVelocitySigma{};
    int16_t m_velocityCorrCoeff{0};

    Vector2<int16_t> m_acceleration{};
    Vector2<uint16_t> m_accelerationSigma{};
    int16_t m_accelerationCorrCoeff{0};

    int16_t m_yawRate{0};
    uint16_t m_yawRateSigma{0};

    uint8_t m_possibleNbOfContourPoints{0};
    std::vector<ContourPointIn2271> m_contourPoints{};
}; // FilteredObjectDataIn2271

//==============================================================================

bool operator==(const microvision::common::sdk::FilteredObjectDataIn2271& lhs,
                const microvision::common::sdk::FilteredObjectDataIn2271& rhs);
bool operator!=(const microvision::common::sdk::FilteredObjectDataIn2271& lhs,
                const microvision::common::sdk::FilteredObjectDataIn2271& rhs);

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
