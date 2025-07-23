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
//! \date Mar 14, 2014
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/measurementlist/special/MeasurementIn2821.hpp>

#include <vector>
#include <iostream>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

class MeasurementSeriesIn2821 final
{
public:
    using MeasurementIn2821 = sdk::MeasurementIn2821;
    using MeasurementVector = std::vector<MeasurementIn2821>;

public:
    MeasurementSeriesIn2821();

public:
    virtual std::streamsize getSerializedSize() const;
    virtual bool deserialize(std::istream& is);
    virtual bool serialize(std::ostream& os) const;

public:
    void addMeasurement(const MeasurementIn2821& meas) { m_measurements.push_back(meas); }

public: // getter
    uint16_t getSize() const { return uint16_t(m_measurements.size()); }

    const MeasurementVector& getMeasurements() const { return m_measurements; }
    MeasurementVector& getMeasurements() { return m_measurements; }

    MeasurementVector::iterator getMeasurement(const MeasurementKeyIn2821 key);
    MeasurementVector::const_iterator getMeasurement(const MeasurementKeyIn2821 key) const;
    bool contains(const MeasurementKeyIn2821 key) const;

public: // setter
    void setMeasurements(const MeasurementVector& measurements) { m_measurements = measurements; }

protected:
    MeasurementVector m_measurements;
}; // MeasurementSeriesIn2821

//==============================================================================

bool operator==(const MeasurementSeriesIn2821& lhs, const MeasurementSeriesIn2821& rhs);
bool operator!=(const MeasurementSeriesIn2821& lhs, const MeasurementSeriesIn2821& rhs);

//==============================================================================

} // namespace sdk
} // namespace common
} // namespace microvision

//==============================================================================
