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
//! \date Jul 16, 2019
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/NtpTime.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Interface implemented by devices that can report the point in time for
//! measurements via a callback method.
//------------------------------------------------------------------------------
class MeasurementTimeProvider
{
public:
    //========================================
    //! \brief The function that is called during packet handling to inform the observer
    //! about the time when the measurement was started.
    //----------------------------------------
    using MeasurementStartCallbackFunction = std::function<
        void(const MeasurementTimeProvider* device, const uint32_t measurementIdx, const NtpTime timestamp)>;

public:
    //========================================
    //! \brief Constructor.
    //----------------------------------------
    MeasurementTimeProvider() = default;

    //========================================
    //! \brief Destructor.
    //----------------------------------------
    virtual ~MeasurementTimeProvider() = default;

public:
    //========================================
    //! \brief Set the call back function for reporting the start of the measurement.
    //!
    //! \param[in] callbackFunction  The new call back function.
    //----------------------------------------
    virtual void setMeasurementStartCallbackFunction(const MeasurementStartCallbackFunction& callbackFunction) = 0;
}; // MeasurementTimeProvider

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
