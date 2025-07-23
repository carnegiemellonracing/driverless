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
//! \date Sep 5, 2013
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/Vector2.hpp>
#include <microvision/common/sdk/NtpTime.hpp>
#include <microvision/common/sdk/ObjectBasic.hpp>

#include <vector>
#include <iostream>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Object with extended tracking information (generic)
//------------------------------------------------------------------------------
class ObjectIn2280 final
{
public:
    enum class Flags : uint16_t
    {
        //! is object tracked using stationary model
        TrackedByStationaryModel = 0x0040U,
        //! Has been detected/validated as mobile. (the current tracking model is irrelevant; this flag just
        //! means it has been moving at some time)
        Mobile = 0x0080U,
        //! Object (stationary or dynamic) has been validated, i.e. valid enough to send out to the interface
        Validated = 0x0100U
    }; // Flags

public:
    ObjectIn2280();
    virtual ~ObjectIn2280() = default;

public:
    virtual std::streamsize getSerializedSize() const;
    virtual bool deserialize(std::istream& is);
    virtual bool serialize(std::ostream& os) const;

public: // getter
    uint16_t getObjectId() const { return m_objectId; }

    uint16_t getFlags() const { return m_flags; }
    bool trackedByStationaryModel() const
    {
        return (m_flags & static_cast<uint16_t>(Flags::TrackedByStationaryModel))
               == static_cast<uint16_t>(Flags::TrackedByStationaryModel);
    }
    bool mobile() const
    {
        return (m_flags & static_cast<uint16_t>(Flags::Mobile)) == static_cast<uint16_t>(Flags::Mobile);
    }
    bool motionModelValidated() const
    {
        return (m_flags & static_cast<uint16_t>(Flags::Validated)) == static_cast<uint16_t>(Flags::Validated);
    }

    uint32_t getObjectAge() const { return m_objectAge; }
    NtpTime getTimestamp() const { return m_timestamp; }
    uint16_t getObjectPredAge() const { return m_objectPredAge; }
    ObjectClass getClassification() const { return m_classification; }
    uint8_t getClassCertainty() const { return m_classCertainty; }
    uint32_t getClassAge() const { return m_classAge; }
    Vector2<float> getReserved0() const { return m_reserved0; }
    Vector2<float> getReserved1() const { return m_reserved1; }
    Vector2<float> getObjBoxCenter() const { return m_objBoxCenter; }
    Vector2<float> getObjBoxCenterSigma() const { return m_objBoxCenterSigma; }
    Vector2<float> getObjBoxSize() const { return m_objBoxSize; }
    Vector2<float> getReserved2() const { return m_reserved2; }
    float getObjBoxOrientation() const { return m_objBoxOrientation; }
    float getObjBoxOrientationSigma() const { return m_objBoxOrientationSigma; }
    Vector2<float> getRelVelocity() const { return m_relVelocity; }
    Vector2<float> getRelVelocitySigma() const { return m_relVelocitySigma; }
    Vector2<float> getAbsVelocity() const { return m_absVelocity; }
    Vector2<float> getAbsVelocitySigma() const { return m_absVelocitySigma; }
    uint64_t getReserved4() const { return m_reserved4; }
    float getReserved5() const { return m_reserved5; }
    float getReserved6() const { return m_reserved6; }
    uint16_t getReserved7() const { return m_reserved7; }
    uint8_t getNbOfContourPoints() const { return uint8_t(m_contourPoints.size()); }
    uint8_t getIdxOfClosestPoint() const { return m_idxOfClosestPoint; }
    RefPointBoxLocation getRefPointLocation() const { return m_refPointLocation; }
    Vector2<float> getRefPointCoords() const { return m_refPointCoords; }
    Vector2<float> getRefPointCoordsSigma() const { return m_refPointCoordsSigma; }
    float getRefPointPosCorrCoeffs() const { return m_refPointPosCorrCoeffs; }
    float getReserved8() const { return m_reserved8; }
    float getReserved9() const { return m_reserved9; }
    uint16_t getObjPriority() const { return m_objPriority; }
    float getObjExtMeasurement() const { return m_objExtMeasurement; }
    const std::vector<Vector2<float>>& getContourPoints() const { return m_contourPoints; }
    std::vector<Vector2<float>>& getContourPoints() { return m_contourPoints; }

public: // setter
    void setObjectId(const uint16_t newObjectId) { m_objectId = newObjectId; }

    void setFlags(const uint16_t newFlags) { m_flags = newFlags; }
    void setTrackedByStationaryModel(const bool set = true)
    {
        m_flags = set ? static_cast<uint16_t>(m_flags | static_cast<uint16_t>(Flags::TrackedByStationaryModel))
                      : static_cast<uint16_t>(m_flags & ~static_cast<uint16_t>(Flags::TrackedByStationaryModel));
    }
    void setMobile(const bool set = true)
    {
        m_flags = set ? static_cast<uint16_t>(m_flags | static_cast<uint16_t>(Flags::Mobile))
                      : static_cast<uint16_t>(m_flags & ~static_cast<uint16_t>(Flags::Mobile));
    }
    void setMotionModelValidated(const bool set = true)
    {
        m_flags = set ? static_cast<uint16_t>(m_flags | static_cast<uint16_t>(Flags::Validated))
                      : static_cast<uint16_t>(m_flags & ~static_cast<uint16_t>(Flags::Validated));
    }

    void setObjectAge(const uint32_t newObjectAge) { m_objectAge = newObjectAge; }
    void setTimestamp(const NtpTime newTimestamp) { m_timestamp = newTimestamp; }
    void setObjectPredAge(const uint16_t newObjectPredAge) { m_objectPredAge = newObjectPredAge; }
    void setClassification(const ObjectClass newClassification) { m_classification = newClassification; }
    void setClassCertainty(const uint8_t newClassCertainty) { m_classCertainty = newClassCertainty; }
    void setClassAge(const uint32_t newClassAge) { m_classAge = newClassAge; }
    void setReserved0(const Vector2<float> reserved0) { m_reserved0 = reserved0; }
    void setReserved1(const Vector2<float> reserved1) { m_reserved1 = reserved1; }
    void setObjBoxCenter(const Vector2<float> newObjBoxCenter) { m_objBoxCenter = newObjBoxCenter; }
    void setObjBoxCenterSigma(const Vector2<float> newObjBoxCenterSigma) { m_objBoxCenterSigma = newObjBoxCenterSigma; }
    void setObjBoxSize(const Vector2<float> newObjBoxSize) { m_objBoxSize = newObjBoxSize; }
    void setReserved2(const Vector2<float> reserved2) { m_reserved2 = reserved2; }

    void setObjBoxOrientation(const float newObjBoxOrientation) { m_objBoxOrientation = newObjBoxOrientation; }
    void setObjBoxOrientationSigma(const float newObjBoxOrientationSigma)
    {
        m_objBoxOrientationSigma = newObjBoxOrientationSigma;
    }
    void setRelVelocity(const Vector2<float> newRelVelocity) { m_relVelocity = newRelVelocity; }
    void setRelVelocitySigma(const Vector2<float> newRelVelocitySigma) { m_relVelocitySigma = newRelVelocitySigma; }
    void setAbsVelocity(const Vector2<float> newAbsVelocity) { m_absVelocity = newAbsVelocity; }
    void setAbsVelocitySigma(const Vector2<float> newAbsVelocitySigma) { m_absVelocitySigma = newAbsVelocitySigma; }
    void setReserved4(const uint64_t reserved4) { m_reserved4 = reserved4; }
    void setReserved5(const float reserved5) { m_reserved5 = reserved5; }
    void setReserved6(const float reserved6) { m_reserved6 = reserved6; }
    void setReserved7(const uint16_t reserved7) { m_reserved7 = reserved7; }
    void setNbOfContourPoints(const uint8_t newNbOfContourPoints) { m_contourPoints.resize(newNbOfContourPoints); }
    void setIdxOfClosestPoint(const uint8_t newIdxOfClosestPoint) { m_idxOfClosestPoint = newIdxOfClosestPoint; }
    void setRefPointLocation(const RefPointBoxLocation newRefPointLocation)
    {
        m_refPointLocation = newRefPointLocation;
    }
    void setRefPointCoords(const Vector2<float> newRefPointCoords) { m_refPointCoords = newRefPointCoords; }
    void setRefPointCoordsSigma(const Vector2<float> newRefPointCoordsSigma)
    {
        m_refPointCoordsSigma = newRefPointCoordsSigma;
    }
    void setRefPointPosCorrCoeffs(const float newRefPointPosCorrCoeffs)
    {
        m_refPointPosCorrCoeffs = newRefPointPosCorrCoeffs;
    }
    void setReserved8(const float reserved8) { m_reserved8 = reserved8; }
    void setReserved9(const float reserved9) { m_reserved9 = reserved9; }
    void setObjPriority(const uint16_t newObjPriority) { m_objPriority = newObjPriority; }
    void setObjExtMeasurement(const float newObjExtMeasurement) { m_objExtMeasurement = newObjExtMeasurement; }
    void setContourPoints(const std::vector<Vector2<float>> contourPoints) { m_contourPoints = contourPoints; }

public:
    Vector2<float> convertRefPoint(const RefPointBoxLocation toPos) const
    {
        return microvision::common::sdk::convertRefPoint(this->getRefPointCoords(),
                                                         this->getObjBoxOrientation(),
                                                         this->getObjBoxSize(),
                                                         this->getRefPointLocation(),
                                                         toPos);
    }

public:
    bool operator==(const ObjectIn2280& other) const;
    bool operator!=(const ObjectIn2280& other) const { return !((*this) == other); }

protected:
    uint16_t m_objectId; //!< ID of this object from tracking.
    uint16_t m_flags; //!< The object flags.
    uint32_t m_objectAge; //!< Number of scans this object has been tracked for.
    NtpTime m_timestamp; //!< Timestamp of the last measurement (COG of Segment) that was used for updating this object.
    uint16_t
        m_objectPredAge; //!< Number of update cycles that this object has only been predicted without measurement updates.
    ObjectClass m_classification; //!< Object classification.

    //!The classification certainty.
    //!
    //! The higher this value is the more reliable is the assigned object class.
    //! Range: 0..100
    uint8_t m_classCertainty;
    uint32_t m_classAge; //!< Time that this object has been classified as current  class in ms. [ms]
    Vector2<float> m_reserved0;
    Vector2<float> m_reserved1;
    Vector2<float> m_objBoxCenter; //!< Center point of this object box.
    Vector2<float> m_objBoxCenterSigma; //!< Standard deviation of the object box center point.
    Vector2<float> m_objBoxSize; //!< Size of the object box.
    Vector2<float> m_reserved2; //!<
    float m_objBoxOrientation; //!< Orientation or heading of the object box [rad].
    float m_objBoxOrientationSigma; //!< Uncertainty of the course angle. [rad]
    Vector2<float>
        m_relVelocity; //!< Velocity of this object in m/s relative to the ego vehicle in the ego vehicle coordinate system.
    Vector2<float> m_relVelocitySigma; //!< Standard deviation of the relative velocity.
    Vector2<float> m_absVelocity; //!< Absolute velocity of this object in m/s.
    Vector2<float> m_absVelocitySigma; //!< Standard deviation of the absolute velocity.

    // 18 bytes reserved/internal
    uint64_t m_reserved4;
    float m_reserved5;
    float m_reserved6;
    uint16_t m_reserved7;

    // nb of contour points  uint8_t
    uint8_t m_idxOfClosestPoint; //!<  Closest contour point of this object as index of the point list.

    RefPointBoxLocation m_refPointLocation; //!< Reference point location.
    Vector2<float> m_refPointCoords; //!< Reference point coordinates. [m]
    Vector2<float> m_refPointCoordsSigma; //!< Standard deviation of the estimated reference point position in m. [m]
    float m_refPointPosCorrCoeffs; //!<  Reference point position correlation coefficient.
    float m_reserved8;
    float m_reserved9;

    //! The mining priority of the object.
    //!
    //! The higher the number, the higher the object priority. Priority is based on (1) motion classification and (2) distance.
    uint16_t m_objPriority;
    float m_objExtMeasurement; //!< Object existence measurement.
    std::vector<Vector2<float>> m_contourPoints; //!< Vector of contour points.
}; // ObjectIn2280

//==============================================================================

} // namespace sdk
} // namespace common
} // namespace microvision

//==============================================================================
