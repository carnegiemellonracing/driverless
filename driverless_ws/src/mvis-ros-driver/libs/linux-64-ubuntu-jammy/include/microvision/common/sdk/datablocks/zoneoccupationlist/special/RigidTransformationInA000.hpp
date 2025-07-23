//==============================================================================
//! \file
//!
//! \brief Data block to store a rigid 3D transformation with position and orientation.
//!
//! $$MICROVISION_LICENSE_BEGIN$$
//! Copyright (c) 2025 MicroVision, Inc., Redmond, U.S.A.
//! All Rights Reserved.
//!
//! For more details, please refer to the accompanying file
//! License.txt.
//! $$MICROVISION_LICENSE_END$$
//!
//! \date Apr 29, 2025
//------------------------------------------------------------------------------

//==============================================================================

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Class representing a rigid 3D transformation with position and orientation.
//!
//! This class stores a 3D transformation consisting of:
//! - 3D position (x, y, z) in millimeters.
//! - 3D orientation (roll, pitch, yaw) in micro radians (Euler angles).
//!
//! The transformation can be used to convert coordinates between different reference frames.
//==============================================================================
class RigidTransformationInA000
{
public:
    //========================================
    //! \brief Get X position in millimeters.
    //! \return X position in millimeters.
    //----------------------------------------
    int32_t getXInMm() const;

    //========================================
    //! \brief Get Y position in millimeters.
    //! \return Y position in millimeters.
    //----------------------------------------
    int32_t getYInMm() const;

    //========================================
    //! \brief Get Z position in millimeters.
    //! \return Z position in millimeters.
    //----------------------------------------
    int32_t getZInMm() const;

    //========================================
    //! \brief Get roll angle in micro radians.
    //! \return Roll angle in micro radians.
    //----------------------------------------
    int32_t getRollInMicroRad() const;

    //========================================
    //! \brief Get pitch angle in micro radians.
    //! \return Pitch angle in micro radians.
    //----------------------------------------
    int32_t getPitchInMicroRad() const;

    //========================================
    //! \brief Get yaw angle in micro radians.
    //! \return Yaw angle in micro radians.
    //----------------------------------------
    int32_t getYawInMicroRad() const;

public:
    //========================================
    //! \brief Set X position in millimeters.
    //! \param[in] xInMm  X position in millimeters.
    //----------------------------------------
    void setXInMm(const int32_t xInMm);

    //========================================
    //! \brief Set Y position in millimeters.
    //! \param[in] yInMm  Y position in millimeters.
    //----------------------------------------
    void setYInMm(const int32_t yInMm);

    //========================================
    //! \brief Set Z position in millimeters.
    //! \param[in] zInMm  Z position in millimeters.
    //----------------------------------------
    void setZInMm(const int32_t zInMm);

    //========================================
    //! \brief Set roll angle in micro radians.
    //! \param[in] rollInMicroRad  Roll angle in micro radians.
    //----------------------------------------
    void setRollInMicroRad(const int32_t rollInMicroRad);

    //========================================
    //! \brief Set pitch angle in micro radians.
    //! \param[in] pitchInMicroRad  Pitch angle in micro radians.
    //----------------------------------------
    void setPitchInMicroRad(const int32_t pitchInMicroRad);

    //========================================
    //! \brief Set yaw angle in micro radians.
    //! \param[in] yawInMicroRad  Yaw angle in micro radians.
    //----------------------------------------
    void setYawInMicroRad(const int32_t yawInMicroRad);

private:
    int32_t m_xInMm{0}; //!< Position x in millimeter.
    int32_t m_yInMm{0}; //!< Position y in millimeter.
    int32_t m_zInMm{0}; //!< Position z in millimeter.
    int32_t m_rollInMicroRad{0}; //!< Roll angle in micro rad.
    int32_t m_pitchInMicroRad{0}; //!< Pitch angle in micro rad.
    int32_t m_yawInMicroRad{0}; //!< Yaw angle in micro rad.
}; // RigidTransformationInA000

//==============================================================================
//! \brief Equality comparison operator for RigidTransformationInA000
//! \param[in] lhs Left-hand side operand
//! \param[in] rhs Right-hand side operand
//! \return True if all transformation parameters are equal, false otherwise
//------------------------------------------------------------------------------
bool operator==(const RigidTransformationInA000& lhs, const RigidTransformationInA000& rhs);

//==============================================================================
//! \brief Inequality comparison operator for RigidTransformationInA000
//! \param[in] lhs Left-hand side operand
//! \param[in] rhs Right-hand side operand
//! \return True if any transformation parameter differs, false if all are equal
//------------------------------------------------------------------------------
inline bool operator!=(const RigidTransformationInA000& lhs, const RigidTransformationInA000& rhs)
{
    return !(lhs == rhs);
}

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
