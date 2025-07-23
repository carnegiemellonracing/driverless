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
//! \date Jul 11, 2016
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

class TimeInterval final
{
public:
    TimeInterval(const NtpTime& startTime,
                 const NtpTime& endTime,
                 const bool startIsIncluded = true,
                 const bool endIsIncluded   = true)
      : m_startTime(startTime), m_endTime(endTime), m_startIsIncluded(startIsIncluded), m_endIsIncluded(endIsIncluded)
    {}

public:
    const NtpTime& getStartTime() const { return m_startTime; }
    const NtpTime& getEndTime() const { return m_endTime; }

    bool isStartTimeIncluded() const { return m_startIsIncluded; }
    bool isEndTimeIncluded() const { return m_endIsIncluded; }

protected:
    NtpTime m_startTime;
    NtpTime m_endTime;
    bool m_startIsIncluded;
    bool m_endIsIncluded;
}; // TimeInterval

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
